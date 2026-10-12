#pragma once

// Warp-level mma.sync decode attention. One CTA = (token row, kv head),
// 4 warps, each warp owning 4 selected pages. Raw K/V/scales are staged via
// cp.async double-buffered per warp so the page i+1 global fetch overlaps
// page i compute. K is decoded on the fly straight into the QK B-fragment
// registers; V is staged dim-major (tv) for the PV B operand. Numerics match
// the production partial+combine contract exactly (per-page E4M3 P at the
// page-max shift, per-page BF16 output rounding, FP32 cross-page merge).
namespace rtp_llm {

struct alignas(128) DecodeWarpShared {
    // K at 72B/token (18-word stride: conflict-free QK reads) and V both
    // double buffered -> the full page period hides both DRAM fetches.
    alignas(128) uint8_t rk[4][2][9216];    // decoded K source, 72B/token rows
    alignas(128) uint8_t rsk[4][2][1024];   // K scales
    alignas(128) uint8_t rv[4][2][9216];    // V packed 8192 | V scale 1024
    alignas(128) uint8_t tv[4][16896];      // decoded V, dim-major E4M3 (132B rows)
    uint8_t pbuf[4][2112];                  // P E4M3, row-major [16,132]
    float mz[4][16][2];
};
static_assert(sizeof(DecodeWarpShared)<=232448,"decode warp kernel exceeds SMEM budget");

__device__ __forceinline__ int sf_off(int g,int token) {
    return (g/4)*512+(token%32)*16+(token/32)*4+g%4;
}

__global__ __launch_bounds__(128,1) void decode_warp_mma(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<DecodeWarpShared*>(smem);
    const int row=blockIdx.x, head=blockIdx.y;
    const int lane=threadIdx.x, warp=lane/32;
    const int ln=lane%32;  // lane within warp for fragment indexing
    const int length=p.rowlens[row];
    const float alpha=p.alpha;

    // Q A-fragments: word w -> row lane/4+8*(w%2), dims (w/4)*32+(w/2%2)*16+(lane%4)*4.
    uint32_t aq[16];
    #pragma unroll
    for(int w=0;w<16;++w) {
        int g_row=ln/4+8*(w%2), dim=(w/4)*32+(w/2%2)*16+(ln%4)*4;
        aq[w]=*reinterpret_cast<const uint32_t*>(p.q+int64_t(row)*p.qn+int64_t(head*16+g_row)*p.qh+dim);
    }

    float o_frag[16][4];
    #pragma unroll
    for(int nt=0;nt<16;++nt) o_frag[nt][0]=o_frag[nt][1]=o_frag[nt][2]=o_frag[nt][3]=0.f;
    float m_run[2]={-CUDART_INF_F,-CUDART_INF_F}, z_run[2]={0.f,0.f};

    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    // 8B cp.async with zero-fill predicate (for the 72B-padded K staging).
    auto cp8=[&](void* dst,const void* src,bool pred) {
        uint32_t d=cute::cast_smem_ptr_to_uint(dst);
        int sz=pred?8:0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                     :: "r"(d),"l"(src),"r"(sz));
    };

    // Resolve this warp's 4 pages up front.
    int physicals[4], visibles[4];
    bool valids[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
        const int slot=warp*4+i;
        int block=p.topk[head*p.th+int64_t(row)*p.tq+slot];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[int64_t(row)*p.tb+block]:-1;
        valid=valid&&physical>=0&&physical<p.pages;
        physicals[i]=physical; valids[i]=valid;
        visibles[i]=valid?length-block*128:0;
    }

    auto issue_k=[&](int buf,int idx) {
        const int phys=physicals[idx];
        const int visible=visibles[idx];
        const uint8_t* kp=p.k+int64_t(phys)*p.kp+head*8192;
        const uint8_t* ksp=p.ks+int64_t(phys)*p.ksp+head*1024;
        // 8B chunks into 72B/token rows (9 chunks per token, last 8B pad).
        #pragma unroll
        for(int c=ln;c<1024;c+=32) {
            int tok=c/8, sub=c%8;
            cp8(s.rk[warp][buf]+tok*72+sub*8,kp+tok*64+sub*8,tok<visible);
        }
        #pragma unroll
        for(int c=ln;c<64;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(ksp+c*16),
                       *reinterpret_cast<uint4*>(s.rsk[warp][buf]+c*16),true);
        cute::cp_async_fence();
    };
    auto issue_v=[&](int buf,int idx) {
        const int phys=physicals[idx];
        const int visible=visibles[idx];
        const uint8_t* vp=p.v+int64_t(phys)*p.vp+head*8192;
        const uint8_t* vsp=p.vs+int64_t(phys)*p.vsp+head*1024;
        uint8_t* rv=s.rv[warp][buf];
        uint8_t* rsv=rv+8192;
        #pragma unroll
        for(int c=ln;c<512;c+=32) {
            bool ok=c/4<visible;
            Copy::copy(*reinterpret_cast<const uint4*>(vp+c*16),
                       *reinterpret_cast<uint4*>(rv+c*16),ok);
        }
        #pragma unroll
        for(int c=ln;c<64;c+=32)
            Copy::copy(*reinterpret_cast<const uint4*>(vsp+c*16),
                       *reinterpret_cast<uint4*>(rsv+c*16),true);
        cute::cp_async_fence();
    };

    // Prologue: fetch page 0 (K and V).
    if(valids[0]) { issue_k(0,0); issue_v(0,0); }

    int mybuf=0;
    #pragma unroll 1
    for(int i=0;i<4;++i) {
        if(!valids[i]) continue;
        // Issue K+V for the next valid page (both get a full page period of
        // overlap), then drain page i's two groups: wait<2> leaves the two
        // newest pending and completes K_i/V_i.
        {
            int next=i+1;
            while(next<4&&!valids[next]) ++next;
            if(next<4) { issue_k(1-mybuf,next); issue_v(1-mybuf,next); cute::cp_async_wait<2>(); }
            else cute::cp_async_wait<0>();
        }
        __syncwarp();
        const int buf=mybuf;
        mybuf=1-mybuf;
        const uint8_t* rk=s.rk[warp][buf];
        const uint8_t* rsk=s.rsk[warp][buf];
        const uint8_t* rv=s.rv[warp][buf];
        const uint8_t* rsv=rv+8192;
        const int visible=visibles[i];

        // V -> tv[dim-major]: lane ln owns dim-group ln (2B lane stride:
        // conflict-free), iterating over the 32 token-quads. Fully unrolled
        // so all 4 rv/rsv loads per iteration issue before the dependent
        // decode4 chain (SMEM latency amortized).
        #pragma unroll
        for(int e=ln;e<1024;e+=32) {
            int q=e%32, tok0=(e/32)*4, g=q*4;
            uint32_t dv[4]={0u,0u,0u,0u};
            #pragma unroll
            for(int j=0;j<4;++j) {
                int tok=tok0+j;
                uint16_t pv=*reinterpret_cast<const uint16_t*>(rv+tok*64+g/2);
                uint8_t sv=rsv[sf_off(g/16,tok)];
                uint32_t dd=decode4(pv,sv);
                const uint8_t* db=reinterpret_cast<const uint8_t*>(&dd);
                #pragma unroll
                for(int b=0;b<4;++b)
                    dv[b]|=uint32_t(db[b])<<(j*8);
            }
            #pragma unroll
            for(int b=0;b<4;++b)
                *reinterpret_cast<uint32_t*>(s.tv[warp]+(g+b)*132+tok0)=dv[b];
        }
        __syncwarp();

        // QK^T: B decoded on the fly from raw K. Per nt, all 8 K/scale
        // loads issue first (independent), then the decode+mma chain runs.
        float sf[16][4];
        #pragma unroll
        for(int nt=0;nt<16;++nt) {
            const int tok=nt*8+ln/4;
            uint16_t kpk[4][2];
            uint8_t skk[4][2];
            #pragma unroll
            for(int ks=0;ks<4;++ks) {
                #pragma unroll
                for(int r=0;r<2;++r) {
                    int dim=ks*32+(ln%4)*4+16*r;
                    kpk[ks][r]=*reinterpret_cast<const uint16_t*>(rk+tok*72+dim/2);
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
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(sf[nt][0]),"+f"(sf[nt][1]),"+f"(sf[nt][2]),"+f"(sf[nt][3])
                    : "r"(aq[ks*4+0]),"r"(aq[ks*4+1]),"r"(aq[ks*4+2]),"r"(aq[ks*4+3]),
                      "r"(bw[0]),"r"(bw[1])
                      : "memory");
            }
        }

        // Row maxima / softmax with production emulated columns.
        float mp[2], den[2];
        {
            float mx0=-CUDART_INF_F,mx1=-CUDART_INF_F;
            #pragma unroll
            for(int nt=0;nt<16;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int tok=nt*8+(ln%4)*2+(e%2);
                    float v=tok<visible?sf[nt][e]:-CUDART_INF_F;
                    if(e<2) mx0=fmaxf(mx0,v); else mx1=fmaxf(mx1,v);
                }
            }
            #pragma unroll
            for(int d=1;d<=2;d<<=1) {
                mx0=fmaxf(mx0,__shfl_xor_sync(0xffffffff,mx0,d));
                mx1=fmaxf(mx1,__shfl_xor_sync(0xffffffff,mx1,d));
            }
            mp[0]=mx0;mp[1]=mx1;
        }
        {
            float sh0=mp[0]==-CUDART_INF_F?0.f:mp[0]*(-alpha);
            float sh1=mp[1]==-CUDART_INF_F?0.f:mp[1]*(-alpha);
            float d0=0.f,d1=0.f;
            #pragma unroll
            for(int nt=0;nt<16;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int tok=nt*8+(ln%4)*2+(e%2);
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
            den[0]=d0;den[1]=d1;
        }

        // Online flash state BEFORE P quantization: m tracks max(mp*alpha);
        // z accumulates den*2^(pm-m) (no double den).
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
        // P -> pbuf E4M3 [16,132], at the page-max shift (production contract).
        #pragma unroll
        for(int nt=0;nt<16;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int g_row=ln/4+8*(e/2), tok=nt*8+(ln%4)*2+(e%2);
                s.pbuf[warp][g_row*132+tok]=F8(sf[nt][e]).raw();
            }
        }
        #pragma unroll
        for(int nt=0;nt<16;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) o_frag[nt][e]*=e<2?aa[0]:aa[1];
        }
        __syncwarp();
        // PV into the retired sf registers (scratch), then fold with the
        // production per-page contract: bf16((P@V)*inv) weighted by den*bb.
        {
            float (&pv)[16][4]=sf;
            #pragma unroll
            for(int nt=0;nt<16;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) pv[nt][e]=0.f;
            }
            #pragma unroll
            for(int nt=0;nt<16;++nt) {
                // All A/B operand loads for this nt issue together before
                // the dependent mma chain (SMEM latency amortized).
                uint32_t av[4][4], bv[4][2];
                #pragma unroll
                for(int ks=0;ks<4;++ks) {
                    #pragma unroll
                    for(int r=0;r<2;++r)
                        bv[ks][r]=*reinterpret_cast<const uint32_t*>(s.tv[warp]+(nt*8+ln/4)*132+ks*32+(ln%4)*4+16*r);
                    av[ks][0]=*reinterpret_cast<const uint32_t*>(s.pbuf[warp]+(ln/4)*132+ks*32+(ln%4)*4);
                    av[ks][1]=*reinterpret_cast<const uint32_t*>(s.pbuf[warp]+(ln/4+8)*132+ks*32+(ln%4)*4);
                    av[ks][2]=*reinterpret_cast<const uint32_t*>(s.pbuf[warp]+(ln/4)*132+ks*32+16+(ln%4)*4);
                    av[ks][3]=*reinterpret_cast<const uint32_t*>(s.pbuf[warp]+(ln/4+8)*132+ks*32+16+(ln%4)*4);
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
            for(int nt=0;nt<16;++nt) {
                #pragma unroll
                for(int e=0;e<4;++e) {
                    int r=e<2?0:1;
                    float rounded=__bfloat162float(__float2bfloat16_rn(pv[nt][e]*invr[r]));
                    o_frag[nt][e]+=rounded*wgt[r];
                }
            }
        }
    }
    cute::cp_async_wait<0>();
    __syncwarp();

    // Cross-warp merge: pack the four 8KB partial-output tiles tightly at
    // the base of the retired raw buffers.
    __syncthreads();
    float(*mo)[16][128]=reinterpret_cast<float(*)[16][128]>(&s.rk[0][0]);
    if(ln%4==0) {
        s.mz[warp][ln/4][0]=m_run[0];s.mz[warp][ln/4][1]=z_run[0];
        s.mz[warp][ln/4+8][0]=m_run[1];s.mz[warp][ln/4+8][1]=z_run[1];
    }
    {
        #pragma unroll
        for(int nt=0;nt<16;++nt) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                int g_row=ln/4+8*(e/2), dim=nt*8+(ln%4)*2+(e%2);
                mo[warp][g_row][dim]=o_frag[nt][e];
            }
        }
    }
    __syncthreads();
    if(warp==0) {
        const int g_row=lane/2, half=lane%2;
        float mt=-CUDART_INF_F;
        #pragma unroll
        for(int w=0;w<4;++w) mt=fmaxf(mt,s.mz[w][g_row][0]);
        float zt=0.f;
        #pragma unroll
        for(int w=0;w<4;++w) {
            float mw=s.mz[w][g_row][0];
            if(mw!=-CUDART_INF_F) zt+=mw==mt?s.mz[w][g_row][1]:s.mz[w][g_row][1]*exp_native(mw-mt);
        }
        float inv=rcp_native(zt>0.f?zt:1.f);
        #pragma unroll 4
        for(int d=0;d<64;++d) {
            int dim=half*64+d;
            float acc=0.f;
            #pragma unroll
            for(int w=0;w<4;++w) {
                float mww=s.mz[w][g_row][0];
                float wgt=mww==-CUDART_INF_F?0.f:(mww==mt?1.f:exp_native(mww-mt));
                acc=fmaf(wgt,mo[w][g_row][dim],acc);
            }
            p.out[(int64_t(row)*64+head*16+g_row)*128+dim]=__float2bfloat16_rn(acc*inv);
        }
        if(half==0) p.lse[int64_t(row)*64+head*16+g_row]=
            zt>0.f?(mt+log_native(zt))*0.6931471805599453f:-CUDART_INF_F;
    }
}

} // namespace rtp_llm
