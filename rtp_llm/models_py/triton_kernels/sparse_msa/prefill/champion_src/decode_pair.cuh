#pragma once

// Pair-split warp-mma decode attention. CTA = (token row, kv head),
// 256 threads = 4 warp pairs; each pair owns 4 selected pages serially
// (slots pair*4..pair*4+3). Within a pair the N dimension is halved: the
// even ("north") warp takes tokens 0-63 for QK and dims 0-63 for PV, the
// odd ("south") warp the rest; softmax max/den cross the pair through a
// tiny SMEM exchange guarded by named barriers. Each warp holds only 8
// fragment tiles (half the registers of the single-warp design). The four
// pairs' partial (O, m, z) states merge once at the end through SMEM.
// Numerics: QK dot products keep the full k-chain per fragment (unchanged);
// den gains one FP32 reassociation across the token halves.
namespace rtp_llm {

struct alignas(128) DecodePairShared {
    alignas(128) uint8_t rk[4][2][4608];       // [pair][side] K tokens, 72B rows (single)
    alignas(128) uint8_t rsk[4][1024];         // shared full K scale plane (swizzled)
    alignas(128) uint8_t rv[4][2][9216];       // full V packed | full V scale plane
    alignas(128) uint8_t tv[4][2][8448];       // per side: decoded V dim-half, 64 rows x 132B
    alignas(128) uint8_t pbuf[4][2112];        // full P E4M3 [16,132] shared by the pair
    float xchg[4][2][32];                      // [pair][side][0..15]=max [16..31]=den
    float mz[8][16][2];                        // per-warp flash state for the merge
};
static_assert(sizeof(DecodePairShared)<=232448,"decode pair kernel exceeds SMEM budget");

__global__ __launch_bounds__(256,1) void decode_pair_mma(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<DecodePairShared*>(smem);
    const int row=blockIdx.x, head=blockIdx.y;
    const int tid=threadIdx.x, warp=tid/32, ln=tid%32;
    const int pair=warp/2, side=warp%2;
    const int bar_id=pair+1;
    const int length=p.rowlens[row];
    const float alpha=p.alpha;

    int physicals[4], visibles[4];
    bool valids[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
        const int slot=pair*4+i;
        int block=p.topk[head*p.th+int64_t(row)*p.tq+slot];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[int64_t(row)*p.tb+block]:-1;
        valid=valid&&physical>=0&&physical<p.pages;
        physicals[i]=physical; valids[i]=valid;
        visibles[i]=valid?length-block*128:0;
    }

    uint32_t aq[16];
    #pragma unroll
    for(int w=0;w<16;++w) {
        int g_row=ln/4+8*(w%2), dim=(w/4)*32+(w/2%2)*16+(ln%4)*4;
        aq[w]=*reinterpret_cast<const uint32_t*>(p.q+int64_t(row)*p.qn+int64_t(head*16+g_row)*p.qh+dim);
    }

    float o_frag[8][4];
    #pragma unroll
    for(int nt=0;nt<8;++nt) o_frag[nt][0]=o_frag[nt][1]=o_frag[nt][2]=o_frag[nt][3]=0.f;
    float m_run[2]={-CUDART_INF_F,-CUDART_INF_F}, z_run[2]={0.f,0.f};

    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    auto cp8=[&](void* dst,const void* src,bool pred) {
        uint32_t d=cute::cast_smem_ptr_to_uint(dst);
        int sz=pred?8:0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                     :: "r"(d),"l"(src),"r"(sz));
    };

    // K: each side fetches its own 64 tokens (72B rows, single buffered,
    // retired after QK); the shared scale plane is copied half per warp.
    // V: each side fetches its 64 tokens + half of the V scale plane into
    // the pair's double-buffered full-V region.
    auto issue_k=[&](int idx) {
        const int phys=physicals[idx];
        const int visible=visibles[idx];
        const int tb=side*64;
        const uint8_t* kp=p.k+int64_t(phys)*p.kp+head*8192+tb*64;
        const uint8_t* ksp=p.ks+int64_t(phys)*p.ksp+head*1024;
        #pragma unroll
        for(int c=ln;c<512;c+=32)
            cp8(s.rk[pair][side]+(c/8)*72+(c%8)*8,kp+(c/8)*64+(c%8)*8,
                tb+c/8<visible);
        #pragma unroll
        for(int c=ln;c<32;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(ksp+(side*32+c)*16),
                       *reinterpret_cast<uint4*>(s.rsk[pair]+(side*32+c)*16),true);
        cute::cp_async_fence();
    };
    auto issue_v=[&](int buf,int idx) {
        const int phys=physicals[idx];
        const int visible=visibles[idx];
        const int tb=side*64;
        const uint8_t* vp=p.v+int64_t(phys)*p.vp+head*8192+tb*64;
        const uint8_t* vsp=p.vs+int64_t(phys)*p.vsp+head*1024;
        #pragma unroll
        for(int c=ln;c<256;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(vp+c*16),
                       *reinterpret_cast<uint4*>(s.rv[pair][buf]+tb*64+c*16),
                       tb+c/4<visible);
        #pragma unroll
        for(int c=ln;c<32;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(vsp+(side*32+c)*16),
                       *reinterpret_cast<uint4*>(s.rv[pair][buf]+8192+(side*32+c)*16),true);
        cute::cp_async_fence();
    };

    if(valids[0]) { issue_k(0); issue_v(0,0); }

    int vbuf=0;
    #pragma unroll 1
    for(int i=0;i<4;++i) {
        if(!valids[i]) continue;
        {
            int next=i+1;
            while(next<4&&!valids[next]) ++next;
            // Issue K+V for page i+1 at the page top: both get a full page
            // of overlap. wait<2> completes page i's K and V groups and
            // leaves the two newest pending.
            if(next<4) { issue_v(1-vbuf,next); cute::cp_async_wait<1>(); }
            else cute::cp_async_wait<0>();
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
        const int buf=vbuf;
        vbuf=1-vbuf;
        const uint8_t* rk=s.rk[pair][side];
        const uint8_t* rsk=s.rsk[pair];
        const uint8_t* rv=s.rv[pair][buf];
        const uint8_t* rsv=rv+8192;
        const int visible=visibles[i];
        const int tok_base=side*64;
        const int dim_base=side*64;

        // Decode this side's dim half of V into tv (dim-major, 132B rows).
        // Fully unrolled: all 16 units' loads are independent and issue
        // back-to-back before the dependent decode4 chains.
        #pragma unroll
        for(int it=0;it<16;++it) {
            int quad=it*2+ln/16;
            int gloc=(ln%16)*4;
            uint32_t dv[4]={0u,0u,0u,0u};
            #pragma unroll
            for(int j=0;j<4;++j) {
                int tok=quad*4+j;
                uint16_t pv=*reinterpret_cast<const uint16_t*>(rv+tok*64+(dim_base+gloc)/2);
                uint8_t sv=rsv[sf_off((dim_base+gloc)/16,tok)];
                uint32_t dd=decode4(pv,sv);
                const uint8_t* db=reinterpret_cast<const uint8_t*>(&dd);
                #pragma unroll
                for(int b=0;b<4;++b)
                    dv[b]|=uint32_t(db[b])<<(j*8);
            }
            #pragma unroll
            for(int b=0;b<4;++b)
                *reinterpret_cast<uint32_t*>(s.tv[pair][side]+(gloc+b)*132+quad*4)=dv[b];
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));

        // QK: S[16 rows, this side's 64 tokens] (8 nt tiles, full k-chain).
        float sf[8][4];
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            const int tok=tok_base+nt*8+ln/4;
            uint16_t kpk[4][2];
            uint8_t skk[4][2];
            #pragma unroll
            for(int ks=0;ks<4;++ks) {
                #pragma unroll
                for(int r=0;r<2;++r) {
                    int dim=ks*32+(ln%4)*4+16*r;
                    kpk[ks][r]=*reinterpret_cast<const uint16_t*>(rk+(tok%64)*72+dim/2);
                    skk[ks][r]=rsk[sf_off(dim/16,tok)];
                }
            }
            sf[nt][0]=sf[nt][1]=sf[nt][2]=sf[nt][3]=0.f;
            #pragma unroll
            for(int ks=0;ks<4;++ks) {
                uint32_t bw[2];
                #pragma unroll
                for(int r=0;r<2;++r)
                    bw[r]=decode4(kpk[ks][r],skk[ks][r]);
                // (mma issued below)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(sf[nt][0]),"+f"(sf[nt][1]),"+f"(sf[nt][2]),"+f"(sf[nt][3])
                    : "r"(aq[ks*4+0]),"r"(aq[ks*4+1]),"r"(aq[ks*4+2]),"r"(aq[ks*4+3]),
                      "r"(bw[0]),"r"(bw[1])
                      : "memory");
            }
        }

        // K staging is retired after QK: prefetch the next valid page's K.
        {
            int next=i+1;
            while(next<4&&!valids[next]) ++next;
            if(next<4) issue_k(next);
        }
        // Local half-max over this side's tokens.
        float mx0=-CUDART_INF_F,mx1=-CUDART_INF_F;
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int tok=tok_base+nt*8+(ln%4)*2+(e%2);
                float v=tok<visible?sf[nt][e]:-CUDART_INF_F;
                if(e<2) mx0=fmaxf(mx0,v); else mx1=fmaxf(mx1,v);
            }
        }
        #pragma unroll
        for(int d=1;d<=2;d<<=1) {
            mx0=fmaxf(mx0,__shfl_xor_sync(0xffffffff,mx0,d));
            mx1=fmaxf(mx1,__shfl_xor_sync(0xffffffff,mx1,d));
        }
        // Exchange half-maxima; both sides then exp at the global page max.
        float mp[2], den[2];
        if(ln%4==0) {
            s.xchg[pair][side][ln/4]=mx0;
            s.xchg[pair][side][ln/4+8]=mx1;
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
        {
            const float* a=s.xchg[pair][side];
            const float* b=s.xchg[pair][1-side];
            mp[0]=fmaxf(a[ln/4],b[ln/4]);
            mp[1]=fmaxf(a[ln/4+8],b[ln/4+8]);
        }
        {
            float sh0=mp[0]==-CUDART_INF_F?0.f:mp[0]*(-alpha);
            float sh1=mp[1]==-CUDART_INF_F?0.f:mp[1]*(-alpha);
            float d0=0.f,d1=0.f;
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int tok=tok_base+nt*8+(ln%4)*2+(e%2);
                    float v=0.f;
                    if(tok<visible) {
                        float ex=fmaf(sf[nt][e],alpha,e<2?sh0:sh1);
                        v=(tok>=32&&tok<96&&tok%16>=12)?exp_emulated(ex):exp_native(ex);
                    }
                    sf[nt][e]=v;
                    if(e<2) d0+=v; else d1+=v;
                }
            }
            #pragma unroll
            for(int d=1;d<=2;d<<=1) {
                d0+=__shfl_xor_sync(0xffffffff,d0,d);
                d1+=__shfl_xor_sync(0xffffffff,d1,d);
            }
            if(ln%4==0) {
                s.xchg[pair][side][16+ln/4]=d0;
                s.xchg[pair][side][16+ln/4+8]=d1;
            }
            asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
            den[0]=d0+s.xchg[pair][1-side][16+ln/4];
            den[1]=d1+s.xchg[pair][1-side][16+ln/4+8];
        }

        // Online flash state (identical on both warps of the pair).
        float nm[2],aa[2],bb[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            float pm=mp[r]==-CUDART_INF_F?-CUDART_INF_F:mp[r]*alpha;
            nm[r]=fmaxf(m_run[r],pm);
            if(nm[r]==-CUDART_INF_F) {
                aa[r]=1.f; bb[r]=0.f;
            } else {
                aa[r]=(z_run[r]>0.f&&m_run[r]!=-CUDART_INF_F)?exp_native(m_run[r]-nm[r]):0.f;
                bb[r]=pm!=-CUDART_INF_F?exp_native(pm-nm[r]):0.f;
                z_run[r]=z_run[r]*aa[r]+den[r]*bb[r];
                m_run[r]=nm[r];
            }
        }
        // P -> shared pbuf: each side writes its token half (page-max shift,
        // production E4M3 contract).
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int g_row=ln/4+8*(e/2), tok=tok_base+nt*8+(ln%4)*2+(e%2);
                s.pbuf[pair][g_row*132+tok]=F8(sf[nt][e]).raw();
            }
        }
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) o_frag[nt][e]*=e<2?aa[0]:aa[1];
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
        // PV into the retired sf registers, then the per-page BF16 contract.
        {
            float (&pv)[8][4]=sf;
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) pv[nt][e]=0.f;
            }
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                uint32_t av[4][4], bv[4][2];
                #pragma unroll
                for(int ks=0;ks<4;++ks) {
                    #pragma unroll
                    for(int r=0;r<2;++r)
                        bv[ks][r]=*reinterpret_cast<const uint32_t*>(s.tv[pair][side]+(nt*8+ln/4)*132+ks*32+(ln%4)*4+16*r);
                    av[ks][0]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4)*132+ks*32+(ln%4)*4);
                    av[ks][1]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4+8)*132+ks*32+(ln%4)*4);
                    av[ks][2]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4)*132+ks*32+16+(ln%4)*4);
                    av[ks][3]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4+8)*132+ks*32+16+(ln%4)*4);
                }
                #pragma unroll
                for(int ks=0;ks<4;++ks)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(pv[nt][0]),"+f"(pv[nt][1]),"+f"(pv[nt][2]),"+f"(pv[nt][3])
                        : "r"(av[ks][0]),"r"(av[ks][1]),"r"(av[ks][2]),"r"(av[ks][3]),
                          "r"(bv[ks][0]),"r"(bv[ks][1])
                          : "memory");
            }
            float invr[2],wgt[2];
            #pragma unroll
            for(int r=0;r<2;++r) {
                invr[r]=rcp_native(den[r]>0.f?den[r]:1.f);
                wgt[r]=den[r]*bb[r];
            }
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int r=e<2?0:1;
                    float rounded=__bfloat162float(__float2bfloat16_rn(pv[nt][e]*invr[r]));
                    o_frag[nt][e]+=rounded*wgt[r];
                }
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
    }
    cute::cp_async_wait<0>();
    asm volatile("bar.sync %0, 64;" :: "r"(bar_id));

    // Cross-pair merge: each pair holds a partial (O over its dim half, m, z)
    // for the same rows. Overlay 8 tiles of [64 dims][17-pad rows] floats on
    // the retired rv buffers; the padded dim-major layout keeps the merge
    // reads conflict-free. Lanes 0..15 own one row each; weights are
    // computed once per lane, not per dim.
    __syncthreads();
    float* mobase=reinterpret_cast<float*>(&s.rv[0][0][0]);
    auto mo=[&](int w,int dim,int g_row)->float& {
        return mobase[w*64*17+dim*17+g_row];
    };
    if(ln%4==0) {
        s.mz[warp][ln/4][0]=m_run[0];s.mz[warp][ln/4][1]=z_run[0];
        s.mz[warp][ln/4+8][0]=m_run[1];s.mz[warp][ln/4+8][1]=z_run[1];
    }
    {
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int g_row=ln/4+8*(e/2), dim=nt*8+(ln%4)*2+(e%2);
                mo(warp,dim,g_row)=o_frag[nt][e];
            }
        }
    }
    __syncthreads();
    {
        // All 32 lanes cooperate per row: lane ln merges dims ln*2 and
        // ln*2+1 and writes one coalesced uint32 (2 bf16) per row.
        const int d0=ln*2;
        #pragma unroll
        for(int g_row=0;g_row<16;++g_row) {
            float mt=-CUDART_INF_F;
            #pragma unroll
            for(int pr=0;pr<4;++pr) mt=fmaxf(mt,s.mz[pr*2+side][g_row][0]);
            float wgt[4]; float zt=0.f;
            #pragma unroll
            for(int pr=0;pr<4;++pr) {
                float mw=s.mz[pr*2+side][g_row][0];
                wgt[pr]=mw==-CUDART_INF_F?0.f:(mw==mt?1.f:exp_native(mw-mt));
                zt+=s.mz[pr*2+side][g_row][1]*wgt[pr];
            }
            float inv=rcp_native(zt>0.f?zt:1.f);
            float a0=0.f,a1=0.f;
            #pragma unroll
            for(int pr=0;pr<4;++pr) {
                a0=fmaf(wgt[pr],mo(pr*2+side,d0,g_row),a0);
                a1=fmaf(wgt[pr],mo(pr*2+side,d0+1,g_row),a1);
            }
            uint32_t pk=normalize_bf16_pair(a0,a1,inv);
            *reinterpret_cast<uint32_t*>(p.out+(int64_t(row)*64+head*16+g_row)*128+side*64+d0)=pk;
            if(side==0&&ln==0)
                p.lse[int64_t(row)*64+head*16+g_row]=
                    zt>0.f?(mt+log_native(zt))*0.6931471805599453f:-CUDART_INF_F;
        }
    }
}

// Page-split variant: grid (rows, 4 heads, G groups), G in {2,4}; group g
// processes pages [g*16/G, (g+1)*16/G) - PP=4/G per warp pair - and writes
// the merged group partial into the production partial ABI (slot g,
// counts=G); the production combine merges the G partials. More CTA fanout
// and shorter chains than the fused kernel: aimed at small batches where
// the fused 16-page chains underfill the GPU. G is chosen so rows*4*G
// stays within one wave (148 SMs).
__global__ __launch_bounds__(256,1) void decode_pair_partial(Params p) {
    const int group=blockIdx.z;
    const int PP=4/gridDim.z;   // pages per pair (2 or 1)
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<DecodePairShared*>(smem);
    const int row=blockIdx.x, head=blockIdx.y;
    const int tid=threadIdx.x, warp=tid/32, ln=tid%32;
    const int pair=warp/2, side=warp%2;
    const int bar_id=pair+1;
    const int length=p.rowlens[row];
    const float alpha=p.alpha;

    int physicals[4], visibles[4];
    bool valids[4];
    #pragma unroll
    for(int i=0;i<4;++i) valids[i]=false;
    for(int i=0;i<PP;++i) {
        const int slot=group*4*PP+pair*PP+i;
        int block=p.topk[head*p.th+int64_t(row)*p.tq+slot];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[int64_t(row)*p.tb+block]:-1;
        valid=valid&&physical>=0&&physical<p.pages;
        physicals[i]=physical; valids[i]=valid;
        visibles[i]=valid?length-block*128:0;
    }

    uint32_t aq[16];
    #pragma unroll
    for(int w=0;w<16;++w) {
        int g_row=ln/4+8*(w%2), dim=(w/4)*32+(w/2%2)*16+(ln%4)*4;
        aq[w]=*reinterpret_cast<const uint32_t*>(p.q+int64_t(row)*p.qn+int64_t(head*16+g_row)*p.qh+dim);
    }

    float o_frag[8][4];
    #pragma unroll
    for(int nt=0;nt<8;++nt) o_frag[nt][0]=o_frag[nt][1]=o_frag[nt][2]=o_frag[nt][3]=0.f;
    float m_run[2]={-CUDART_INF_F,-CUDART_INF_F}, z_run[2]={0.f,0.f};

    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    auto cp8=[&](void* dst,const void* src,bool pred) {
        uint32_t d=cute::cast_smem_ptr_to_uint(dst);
        int sz=pred?8:0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                     :: "r"(d),"l"(src),"r"(sz));
    };

    // K: each side fetches its own 64 tokens (72B rows, single buffered,
    // retired after QK); the shared scale plane is copied half per warp.
    // V: each side fetches its 64 tokens + half of the V scale plane into
    // the pair's double-buffered full-V region.
    auto issue_k=[&](int idx) {
        const int phys=physicals[idx];
        const int visible=visibles[idx];
        const int tb=side*64;
        const uint8_t* kp=p.k+int64_t(phys)*p.kp+head*8192+tb*64;
        const uint8_t* ksp=p.ks+int64_t(phys)*p.ksp+head*1024;
        #pragma unroll
        for(int c=ln;c<512;c+=32)
            cp8(s.rk[pair][side]+(c/8)*72+(c%8)*8,kp+(c/8)*64+(c%8)*8,
                tb+c/8<visible);
        #pragma unroll
        for(int c=ln;c<32;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(ksp+(side*32+c)*16),
                       *reinterpret_cast<uint4*>(s.rsk[pair]+(side*32+c)*16),true);
        cute::cp_async_fence();
    };
    auto issue_v=[&](int buf,int idx) {
        const int phys=physicals[idx];
        const int visible=visibles[idx];
        const int tb=side*64;
        const uint8_t* vp=p.v+int64_t(phys)*p.vp+head*8192+tb*64;
        const uint8_t* vsp=p.vs+int64_t(phys)*p.vsp+head*1024;
        #pragma unroll
        for(int c=ln;c<256;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(vp+c*16),
                       *reinterpret_cast<uint4*>(s.rv[pair][buf]+tb*64+c*16),
                       tb+c/4<visible);
        #pragma unroll
        for(int c=ln;c<32;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(vsp+(side*32+c)*16),
                       *reinterpret_cast<uint4*>(s.rv[pair][buf]+8192+(side*32+c)*16),true);
        cute::cp_async_fence();
    };

    if(valids[0]) { issue_k(0); issue_v(0,0); }

    int vbuf=0;
    #pragma unroll 1
    for(int i=0;i<PP;++i) {
        if(!valids[i]) continue;
        {
            int next=i+1;
            while(next<4&&!valids[next]) ++next;
            if(next<4) { issue_v(1-vbuf,next); cute::cp_async_wait<1>(); }
            else cute::cp_async_wait<0>();
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
        const int buf=vbuf;
        vbuf=1-vbuf;
        const uint8_t* rk=s.rk[pair][side];
        const uint8_t* rsk=s.rsk[pair];
        const uint8_t* rv=s.rv[pair][buf];
        const uint8_t* rsv=rv+8192;
        const int visible=visibles[i];
        const int tok_base=side*64;
        const int dim_base=side*64;

        // Decode this side's dim half of V into tv (dim-major, 132B rows).
        // Fully unrolled: all 16 units' loads are independent and issue
        // back-to-back before the dependent decode4 chains.
        #pragma unroll
        for(int it=0;it<16;++it) {
            int quad=it*2+ln/16;
            int gloc=(ln%16)*4;
            uint32_t dv[4]={0u,0u,0u,0u};
            #pragma unroll
            for(int j=0;j<4;++j) {
                int tok=quad*4+j;
                uint16_t pv=*reinterpret_cast<const uint16_t*>(rv+tok*64+(dim_base+gloc)/2);
                uint8_t sv=rsv[sf_off((dim_base+gloc)/16,tok)];
                uint32_t dd=decode4(pv,sv);
                const uint8_t* db=reinterpret_cast<const uint8_t*>(&dd);
                #pragma unroll
                for(int b=0;b<4;++b)
                    dv[b]|=uint32_t(db[b])<<(j*8);
            }
            #pragma unroll
            for(int b=0;b<4;++b)
                *reinterpret_cast<uint32_t*>(s.tv[pair][side]+(gloc+b)*132+quad*4)=dv[b];
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));

        // QK: S[16 rows, this side's 64 tokens] (8 nt tiles, full k-chain).
        float sf[8][4];
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            const int tok=tok_base+nt*8+ln/4;
            uint16_t kpk[4][2];
            uint8_t skk[4][2];
            #pragma unroll
            for(int ks=0;ks<4;++ks) {
                #pragma unroll
                for(int r=0;r<2;++r) {
                    int dim=ks*32+(ln%4)*4+16*r;
                    kpk[ks][r]=*reinterpret_cast<const uint16_t*>(rk+(tok%64)*72+dim/2);
                    skk[ks][r]=rsk[sf_off(dim/16,tok)];
                }
            }
            sf[nt][0]=sf[nt][1]=sf[nt][2]=sf[nt][3]=0.f;
            #pragma unroll
            for(int ks=0;ks<4;++ks) {
                uint32_t bw[2];
                #pragma unroll
                for(int r=0;r<2;++r)
                    bw[r]=decode4(kpk[ks][r],skk[ks][r]);
                // (mma issued below)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(sf[nt][0]),"+f"(sf[nt][1]),"+f"(sf[nt][2]),"+f"(sf[nt][3])
                    : "r"(aq[ks*4+0]),"r"(aq[ks*4+1]),"r"(aq[ks*4+2]),"r"(aq[ks*4+3]),
                      "r"(bw[0]),"r"(bw[1])
                      : "memory");
            }
        }

        // K staging retired after QK: prefetch the next page's K data+scale
        // K staging is retired after QK: prefetch the next valid page's K.
        {
            int next=i+1;
            while(next<PP&&!valids[next]) ++next;
            if(next<PP) issue_k(next);
        }
        // Local half-max over this side's tokens.
        float mx0=-CUDART_INF_F,mx1=-CUDART_INF_F;
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int tok=tok_base+nt*8+(ln%4)*2+(e%2);
                float v=tok<visible?sf[nt][e]:-CUDART_INF_F;
                if(e<2) mx0=fmaxf(mx0,v); else mx1=fmaxf(mx1,v);
            }
        }
        #pragma unroll
        for(int d=1;d<=2;d<<=1) {
            mx0=fmaxf(mx0,__shfl_xor_sync(0xffffffff,mx0,d));
            mx1=fmaxf(mx1,__shfl_xor_sync(0xffffffff,mx1,d));
        }
        // Exchange half-maxima; both sides then exp at the global page max.
        float mp[2], den[2];
        if(ln%4==0) {
            s.xchg[pair][side][ln/4]=mx0;
            s.xchg[pair][side][ln/4+8]=mx1;
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
        {
            const float* a=s.xchg[pair][side];
            const float* b=s.xchg[pair][1-side];
            mp[0]=fmaxf(a[ln/4],b[ln/4]);
            mp[1]=fmaxf(a[ln/4+8],b[ln/4+8]);
        }
        {
            float sh0=mp[0]==-CUDART_INF_F?0.f:mp[0]*(-alpha);
            float sh1=mp[1]==-CUDART_INF_F?0.f:mp[1]*(-alpha);
            float d0=0.f,d1=0.f;
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int tok=tok_base+nt*8+(ln%4)*2+(e%2);
                    float v=0.f;
                    if(tok<visible) {
                        float ex=fmaf(sf[nt][e],alpha,e<2?sh0:sh1);
                        v=(tok>=32&&tok<96&&tok%16>=12)?exp_emulated(ex):exp_native(ex);
                    }
                    sf[nt][e]=v;
                    if(e<2) d0+=v; else d1+=v;
                }
            }
            #pragma unroll
            for(int d=1;d<=2;d<<=1) {
                d0+=__shfl_xor_sync(0xffffffff,d0,d);
                d1+=__shfl_xor_sync(0xffffffff,d1,d);
            }
            if(ln%4==0) {
                s.xchg[pair][side][16+ln/4]=d0;
                s.xchg[pair][side][16+ln/4+8]=d1;
            }
            asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
            den[0]=d0+s.xchg[pair][1-side][16+ln/4];
            den[1]=d1+s.xchg[pair][1-side][16+ln/4+8];
        }

        // Online flash state (identical on both warps of the pair).
        float nm[2],aa[2],bb[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            float pm=mp[r]==-CUDART_INF_F?-CUDART_INF_F:mp[r]*alpha;
            nm[r]=fmaxf(m_run[r],pm);
            if(nm[r]==-CUDART_INF_F) {
                aa[r]=1.f; bb[r]=0.f;
            } else {
                aa[r]=(z_run[r]>0.f&&m_run[r]!=-CUDART_INF_F)?exp_native(m_run[r]-nm[r]):0.f;
                bb[r]=pm!=-CUDART_INF_F?exp_native(pm-nm[r]):0.f;
                z_run[r]=z_run[r]*aa[r]+den[r]*bb[r];
                m_run[r]=nm[r];
            }
        }
        // P -> shared pbuf: each side writes its token half (page-max shift,
        // production E4M3 contract).
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int g_row=ln/4+8*(e/2), tok=tok_base+nt*8+(ln%4)*2+(e%2);
                s.pbuf[pair][g_row*132+tok]=F8(sf[nt][e]).raw();
            }
        }
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) o_frag[nt][e]*=e<2?aa[0]:aa[1];
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
        // PV into the retired sf registers, then the per-page BF16 contract.
        {
            float (&pv)[8][4]=sf;
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) pv[nt][e]=0.f;
            }
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                uint32_t av[4][4], bv[4][2];
                #pragma unroll
                for(int ks=0;ks<4;++ks) {
                    #pragma unroll
                    for(int r=0;r<2;++r)
                        bv[ks][r]=*reinterpret_cast<const uint32_t*>(s.tv[pair][side]+(nt*8+ln/4)*132+ks*32+(ln%4)*4+16*r);
                    av[ks][0]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4)*132+ks*32+(ln%4)*4);
                    av[ks][1]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4+8)*132+ks*32+(ln%4)*4);
                    av[ks][2]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4)*132+ks*32+16+(ln%4)*4);
                    av[ks][3]=*reinterpret_cast<const uint32_t*>(s.pbuf[pair]+(ln/4+8)*132+ks*32+16+(ln%4)*4);
                }
                #pragma unroll
                for(int ks=0;ks<4;++ks)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(pv[nt][0]),"+f"(pv[nt][1]),"+f"(pv[nt][2]),"+f"(pv[nt][3])
                        : "r"(av[ks][0]),"r"(av[ks][1]),"r"(av[ks][2]),"r"(av[ks][3]),
                          "r"(bv[ks][0]),"r"(bv[ks][1])
                          : "memory");
            }
            float invr[2],wgt[2];
            #pragma unroll
            for(int r=0;r<2;++r) {
                invr[r]=rcp_native(den[r]>0.f?den[r]:1.f);
                wgt[r]=den[r]*bb[r];
            }
            #pragma unroll
            for(int nt=0;nt<8;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int r=e<2?0:1;
                    float rounded=__bfloat162float(__float2bfloat16_rn(pv[nt][e]*invr[r]));
                    o_frag[nt][e]+=rounded*wgt[r];
                }
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(bar_id));
    }
    cute::cp_async_wait<0>();
    asm volatile("bar.sync %0, 64;" :: "r"(bar_id));

    // Cross-pair merge: each pair holds a partial (O over its dim half, m, z)
    // for the same rows. Overlay 8 tiles of [64 dims][17-pad rows] floats on
    // the retired rv buffers; the padded dim-major layout keeps the merge
    // reads conflict-free. Lanes 0..15 own one row each; weights are
    // computed once per lane, not per dim.
    __syncthreads();
    float* mobase=reinterpret_cast<float*>(&s.rv[0][0][0]);
    auto mo=[&](int w,int dim,int g_row)->float& {
        return mobase[w*64*17+dim*17+g_row];
    };
    if(ln%4==0) {
        s.mz[warp][ln/4][0]=m_run[0];s.mz[warp][ln/4][1]=z_run[0];
        s.mz[warp][ln/4+8][0]=m_run[1];s.mz[warp][ln/4+8][1]=z_run[1];
    }
    {
        #pragma unroll
        for(int nt=0;nt<8;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int g_row=ln/4+8*(e/2), dim=nt*8+(ln%4)*2+(e%2);
                mo(warp,dim,g_row)=o_frag[nt][e];
            }
        }
    }
    __syncthreads();
    {
        // All 32 lanes cooperate per row: lane ln merges dims ln*2 and
        // ln*2+1 and writes one coalesced uint32 (2 bf16) per row.
        const int d0=ln*2;
        #pragma unroll
        for(int g_row=0;g_row<16;++g_row) {
            float mt=-CUDART_INF_F;
            #pragma unroll
            for(int pr=0;pr<4;++pr) mt=fmaxf(mt,s.mz[pr*2+side][g_row][0]);
            float wgt[4]; float zt=0.f;
            #pragma unroll
            for(int pr=0;pr<4;++pr) {
                float mw=s.mz[pr*2+side][g_row][0];
                wgt[pr]=mw==-CUDART_INF_F?0.f:(mw==mt?1.f:exp_native(mw-mt));
                zt+=s.mz[pr*2+side][g_row][1]*wgt[pr];
            }
            float inv=rcp_native(zt>0.f?zt:1.f);
            float a0=0.f,a1=0.f;
            #pragma unroll
            for(int pr=0;pr<4;++pr) {
                a0=fmaf(wgt[pr],mo(pr*2+side,d0,g_row),a0);
                a1=fmaf(wgt[pr],mo(pr*2+side,d0+1,g_row),a1);
            }
            uint32_t pk=normalize_bf16_pair(a0,a1,inv);
            const int64_t base=((int64_t(row)*4+head)*16+group)*16+g_row;
            *reinterpret_cast<uint32_t*>(p.partial+base*128+side*64+d0)=pk;
            if(side==0&&ln==0)
                p.partial_lse[base]=
                    zt>0.f?(mt+log_native(zt))*0.6931471805599453f:-CUDART_INF_F;
        }
        if(group==0&&tid==0)
            p.counts[int64_t(row)*4+head]=gridDim.z;
    }

}

} // namespace rtp_llm
