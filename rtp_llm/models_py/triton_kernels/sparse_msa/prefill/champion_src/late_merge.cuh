#pragma once

namespace rtp_llm {
using Layout64=decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<F8>{},cute::Shape<cute::_64,cute::_128>{}));
using OutputLayout64=decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},cute::Shape<cute::_64,cute::_128>{}));

struct alignas(128) LateShared {
    alignas(128) uint8_t q[8192],k[2][16384],v[2][16384],p[2][8192];
    alignas(128) uint32_t partial[8][128][32];
    float lp[16][64],inv[2][64];
    alignas(8) uint64_t kv_ready[2],kv_free[2],p_ready[2],p_free[2],stats_free[2];
    uint64_t score_ready,score_free,o_ready,o_free,done;
    uint32_t tmem;
};
static_assert(sizeof(LateShared)<=232448,"late merge exceeds this device's SMEM budget");

template<size_t... I>
__device__ __forceinline__ void load64_m64_impl(uint32_t addr,float* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_16dp32b64x::copy(addr,reinterpret_cast<uint32_t*>(v)[I]...);
}
__device__ __forceinline__ void load64_m64(uint32_t addr,float* v) {
    load64_m64_impl(addr,v,std::make_index_sequence<64>{});
}
__device__ __forceinline__ uint64_t descriptor64(uint8_t* ptr) {
    return cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
        cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(ptr)),Layout64{}));
}
__device__ __forceinline__ uint32_t normalize_bf16_pair(float a,float b,float inv) {
    uint32_t r;
    asm("{ .reg .b64 x,s; .reg .b32 lo,hi; mov.b64 x,{%1,%2}; mov.b64 s,{%3,%3}; "
        "mul.rn.f32x2 x,x,s; mov.b64 {lo,hi},x; cvt.rn.bf16x2.f32 %0,hi,lo; }"
        : "=r"(r):"f"(a),"f"(b),"f"(inv));
    return r;
}
__device__ __forceinline__ void accumulate_bf16_pair(float& a,float& b,uint32_t packed,float weight) {
    asm("{ .reg .b32 lo,hi; .reg .b64 x,y,w,r; shl.b32 lo,%2,16; and.b32 hi,%2,0xffff0000; "
        "mov.b64 x,{lo,hi}; mov.b64 y,{%0,%1}; mov.b64 w,{%3,%3}; "
        "mul.rn.f32x2 x,x,w; add.rn.f32x2 r,y,x; mov.b64 {%0,%1},r; }"
        : "+f"(a),"+f"(b):"r"(packed),"f"(weight));
}

__global__ __launch_bounds__(512,1) void grouped_cutlass_late(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<LateShared*>(smem);
    const int tid=threadIdx.x,wg=tid/128,local=tid%128,warp=local/32,lane=local%32;
    const int row=warp*16+lane%16,half=lane/16;
    const int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1];
    const int qbase=start+int(blockIdx.x)*4;
    if(qbase>=stop) return;
    const int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.stats_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.score_free,1);
        cutlass::arch::ClusterBarrier::init(&s.o_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.o_free,1);
        cutlass::arch::ClusterBarrier::init(&s.done,1);
    }
    if(tid<32) {
        cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    for(int i=tid;i<2048;i+=512) {
        int qr=i/32,d=i%32*4,q=qbase+qr/16;
        uint32_t v=q<stop?*reinterpret_cast<const uint32_t*>(p.q+int64_t(q)*p.qn+(head*16+qr%16)*p.qh+d):0;
        *reinterpret_cast<uint32_t*>(s.q+Layout64{}(cute::make_coord(qr,d)))=v;
    }
    cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    const uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if(wg==0) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        int query=qbase+warp,limit=query-start+length-(stop-start);
        for(int j=0;j<16;++j) {
            int slot=j%2;
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,j%2);after_sync();
            float v[64];load64_m64(t+addr,v);cutlass::arch::fence_view_async_tmem_load();
            before_sync();wg_sync<1>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.score_free);
            int block=p.topk[head*p.th+int64_t(qbase)*p.tq+j];
            bool valid=block>=0&&block<p.max_pages&&block*128<length;
            int page=valid?p.table[batch*p.tb+block]:-1;
            valid=valid&&page>=0&&page<p.pages&&query<stop;
            int last=valid?min(length-1,limit)-block*128:-1;
            float maxima[8];
            #pragma unroll
            for(int c=0;c<8;++c) maxima[c]=-CUDART_INF_F;
            #pragma unroll
            for(int c=0;c<64;++c) {
                float masked;
                asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                    : "=f"(masked):"f"(v[c]),"f"(-CUDART_INF_F),"r"(half*64+c),"r"(last));
                v[c]=masked;maxima[c%8]=max_native(maxima[c%8],masked);
            }
            float maximum=max_native(max_native(max_native(maxima[0],maxima[1]),max_native(maxima[2],maxima[3])),
                                     max_native(max_native(maxima[4],maxima[5]),max_native(maxima[6],maxima[7])));
            maximum=max_native(maximum,__shfl_xor_sync(0xffffffff,maximum,16));
            float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
            #pragma unroll
            for(int c=0;c<64;c+=2) {
                affine_pair(v[c],v[c+1],p.alpha,shift);
                float a=exp_native(v[c]),b=exp_native(v[c+1]);
                if(c%16>=12) {
                    float ea=exp_emulated(v[c]),eb=exp_emulated(v[c+1]);
                    int em=(c<32)?half:1-half;
                    asm("{ .reg .pred sel; setp.ne.u32 sel,%4,0; selp.f32 %0,%2,%0,sel; selp.f32 %1,%3,%1,sel; }"
                        : "+f"(a),"+f"(b):"f"(ea),"f"(eb),"r"(em));
                }
                v[c]=a;v[c+1]=b;
            }
            float sums[8]={};
            #pragma unroll
            for(int i=0;i<8;++i) {
                #pragma unroll
                for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[i*8+c],v[i*8+c+1]);
            }
            #pragma unroll
            for(int c=0;c<8;++c) sums[c]=__shfl_sync(0xffffffff,sums[c],lane%16);
            #pragma unroll
            for(int i=0;i<8;++i) {
                #pragma unroll
                for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[i*8+c],v[i*8+c+1]);
            }
            float den=((sums[0]+sums[2])+(sums[4]+sums[6]))+((sums[1]+sums[3])+(sums[5]+sums[7]));
            den=__shfl_sync(0xffffffff,den,lane%16+16);
            if(j>=2) {
                cutlass::arch::ClusterBarrier::wait(&s.p_free[slot],((j/2)-1)%2);
                cutlass::arch::ClusterBarrier::wait(&s.stats_free[slot],((j/2)-1)%2);
            }
            #pragma unroll
            for(int c=0;c<64;c+=4) {
                uint32_t packed=pack_probability(v[c],v[c+1],v[c+2],v[c+3]);
                *reinterpret_cast<uint32_t*>(s.p[slot]+Layout64{}(cute::make_coord(row,half*64+c)))=packed;
            }
            if(half==0) {
                s.inv[slot][row]=rcp_native(den>0.f?den:1.f);
                s.lp[j][row]=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
            }
            cutlass::arch::fence_view_async_shared();wg_sync<1>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready[slot]);
        }
    } else if(wg==1) {
        cutlass::arch::warpgroup_reg_alloc<192>();
        for(int j=0;j<16;++j) {
            cutlass::arch::ClusterBarrier::wait(&s.o_ready,j%2);after_sync();
            float values[64];load64_m64(t+128+addr,values);cutlass::arch::fence_view_async_tmem_load();
            float inv=s.inv[j%2][row];
            wg_sync<2>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.stats_free[j%2]);
            uint32_t packed[32];
            #pragma unroll
            for(int c=0;c<32;++c) packed[c]=normalize_bf16_pair(values[c*2],values[c*2+1],inv);
            if(j<8) {
                store32(t+256+j*32+addr,packed);cutlass::arch::fence_view_async_tmem_store();
            } else {
                #pragma unroll
                for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(&s.partial[j-8][local][c^((local&7)*4)])=*reinterpret_cast<uint4*>(packed+c);
            }
            before_sync();wg_sync<2>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.o_free);
        }
        float weights[16],maximum=-CUDART_INF_F,total=0.f;
        #pragma unroll
        for(int j=0;j<16;++j) {weights[j]=s.lp[j][row];maximum=max_native(maximum,weights[j]);}
        float safe=maximum==-CUDART_INF_F?0.f:maximum;
        #pragma unroll
        for(int j=0;j<16;++j) {weights[j]=exp_native(weights[j]-safe);total+=weights[j];}
        float accum[64]={};
        for(int j=0;j<16;++j) {
            uint32_t packed[32];
            if(j<8) {
                load32(t+256+j*32+addr,reinterpret_cast<float*>(packed));cutlass::arch::fence_view_async_tmem_load();
            } else {
                #pragma unroll
                for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(packed+c)=*reinterpret_cast<uint4*>(&s.partial[j-8][local][c^((local&7)*4)]);
            }
            #pragma unroll
            for(int c=0;c<32;++c) accumulate_bf16_pair(accum[c*2],accum[c*2+1],packed[c],weights[j]);
        }
        float inv=rcp_native(total>0.f?total:1.f);
        auto* out=reinterpret_cast<__nv_bfloat16*>(s.p);
        #pragma unroll
        for(int c=0;c<64;c+=2) {
            *reinterpret_cast<uint32_t*>(out+OutputLayout64{}(cute::make_coord(row,half*64+c)))=normalize_bf16_pair(accum[c],accum[c+1],inv);
        }
        int query=qbase+warp;
        if(half==0&&query<stop) p.lse[int64_t(query)*64+head*16+lane%16]=total>0.f?(maximum+log_native(total))*0.6931471805599453f:-CUDART_INF_F;
        wg_sync<2>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else if(wg==2) {
        cutlass::arch::warpgroup_reg_dealloc<96>();
        using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
        for(int j=0;j<16;++j) {
            int slot=j%2;
            if(j>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((j/2)-1)%2);
            int block=p.topk[head*p.th+int64_t(qbase)*p.tq+j];
            bool valid=block>=0&&block<p.max_pages&&block*128<length;
            int page=valid?p.table[batch*p.tb+block]:-1;
            valid=valid&&page>=0&&page<p.pages;
            int safe_page=valid?page:0;
            #pragma unroll
            for(int i=0;i<8;++i) {
                int token=local/8+i*16,d=local%8*16;
                int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                bool present=valid&&block*128+token<length;
                Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.k[slot]+Layout{}(cute::make_coord(token,d))),present);
                Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.v[slot]+VLayout{}(cute::make_coord(d,token))),present);
            }
            cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_view_async_shared();wg_sync<3>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<80>();
        if(warp==0) {
            uint64_t dq=descriptor64(s.q);
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            for(int j=0;j<17;++j) {
                if(j<16) {
                    int slot=j%2;
                    if(j>0) cutlass::arch::ClusterBarrier::wait(&s.score_free,(j-1)%2);
                    cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(j/2)%2);after_sync();
                    uint64_t dk=descriptor(s.k[slot]);
                    #pragma unroll
                    for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                    cutlass::arch::umma_arrive(&s.score_ready);
                }
                if(j>0) {
                    int page=j-1,slot=page%2;
                    cutlass::arch::ClusterBarrier::wait(&s.p_ready[slot],(page/2)%2);
                    if(page>0) cutlass::arch::ClusterBarrier::wait(&s.o_free,(page-1)%2);
                    after_sync();
                    uint64_t dp=descriptor64(s.p[slot]);
                    uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.v[slot])),VLayout{}));
                    #pragma unroll
                    for(int k=0;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+128,k!=0,inst_pv);
                    cutlass::arch::umma_arrive(&s.o_ready);
                    cutlass::arch::umma_arrive(&s.p_free[slot]);
                    cutlass::arch::umma_arrive(&s.kv_free[slot]);
                }
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.p);
    for(int i=tid;i<1024;i+=512) {
        int r=i/16,d=i%16*8,q=qbase+r/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+r%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout64{}(cute::make_coord(r,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,512);
}
} // namespace rtp_llm
