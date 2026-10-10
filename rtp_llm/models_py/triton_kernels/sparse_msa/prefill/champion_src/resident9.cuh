#pragma once

namespace rtp_llm {
struct alignas(128) Resident9Shared {
    alignas(128) uint8_t q[16384],p[16384],kv[4][16384];
    int blocks[16],physical[16];
    alignas(8) uint64_t kv_ready[2],kv_free[2],score_ready,pv_ready,math_free,p_half,p_ready,done;
    uint32_t tmem;
};
static_assert(sizeof(Resident9Shared)<=232448,"resident9 kernel exceeds SMEM budget");

__global__ __launch_bounds__(256,2) void grouped_cutlass_resident9(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident9Shared*>(smem);
    int tid=threadIdx.x,local=tid%128,warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);cutlass::arch::ClusterBarrier::init(&s.pv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.math_free,1);
        cutlass::arch::ClusterBarrier::init(&s.p_half,1);cutlass::arch::ClusterBarrier::init(&s.p_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.done,1);
    }
    if(tid<16) {
        int block=p.topk[head*p.th+int64_t(qbase)*p.tq+tid];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[batch*p.tb+block]:-1;
        s.blocks[tid]=block;s.physical[tid]=(valid&&physical>=0&&physical<p.pages)?physical:-1;
    }
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    for(int i=tid;i<1024;i+=256) {
        int row=i/8,d=i%8*16,q=qbase+row/16,safe_q=q<stop?q:start;
        Copy::copy(*reinterpret_cast<const uint4*>(p.q+int64_t(safe_q)*p.qn+(head*16+row%16)*p.qh+d),
                   *reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(row,d))),q<stop);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if(tid<128) {
        cutlass::arch::warpgroup_reg_alloc<192>();
        int query=qbase+local/16,limit=query-start+length-(stop-start);
        float running[128];
        #pragma unroll
        for(int i=0;i<128;++i) running[i]=0.f;
        float m=-CUDART_INF_F,z=0.f;
        const int col=local;
        for(int page=0;page<16;++page) {
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,page%2);after_sync();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0&&query<stop;
            int last=valid?min(length-1,limit)-block*128:-1;
            bool full=__all_sync(0xffffffff,last>=127);
            float sc[32],tm[4];
            float maximum;
            if(full) {
                #pragma unroll
                for(int c0=0;c0<128;c0+=32) load32max(t+addr+c0,sc,tm[c0/32]);
                cutlass::arch::fence_view_async_tmem_load();
                maximum=max_native(max_native(tm[0],tm[1]),max_native(tm[2],tm[3]));
            } else {
                float maxima[8];
                #pragma unroll
                for(int i=0;i<8;++i) maxima[i]=-CUDART_INF_F;
                #pragma unroll
                for(int c0=0;c0<128;c0+=32) {
                    load32(t+addr+c0,sc);
                    cutlass::arch::fence_view_async_tmem_load();
                    #pragma unroll
                    for(int c=0;c<32;++c) {
                        float masked;
                        asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                            : "=f"(masked) : "f"(sc[c]),"f"(-CUDART_INF_F),"r"(c0+c),"r"(last));
                        maxima[c%8]=max_native(maxima[c%8],masked);
                    }
                }
                maximum=max_native(max_native(max_native(maxima[0],maxima[1]),max_native(maxima[2],maxima[3])),
                                   max_native(max_native(maxima[4],maxima[5]),max_native(maxima[6],maxima[7])));
            }
            float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
            float sums[8]={};
            #pragma unroll
            for(int c0=0;c0<128;c0+=32) {
                load32(t+addr+c0,sc);
                cutlass::arch::fence_view_async_tmem_load();
                uint32_t packed[8];
                #pragma unroll
                for(int g=0;g<4;++g) {
                    float e[8];
                    #pragma unroll
                    for(int i=0;i<8;i+=2) {
                        float x=sc[g*8+i],y=sc[g*8+i+1];
                        if(!full) {
                            float mx,my;
                            asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                                : "=f"(mx) : "f"(sc[g*8+i]),"f"(-CUDART_INF_F),"r"(c0+g*8+i),"r"(last));
                            asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                                : "=f"(my) : "f"(sc[g*8+i+1]),"f"(-CUDART_INF_F),"r"(c0+g*8+i+1),"r"(last));
                            x=mx;y=my;
                        }
                        affine_pair(x,y,p.alpha,shift);
                        const int gc=c0+g*8+i;
                        x=(gc>=32&&gc<96&&gc%16>=12)?exp_emulated(x):exp_native(x);
                        y=(gc+1>=32&&gc+1<96&&(gc+1)%16>=12)?exp_emulated(y):exp_native(y);
                        e[i]=x;e[i+1]=y;
                        add_pair(sums[i],sums[i+1],x,y);
                    }
                    packed[2*g]=pack_probability(e[0],e[1],e[2],e[3]);
                    packed[2*g+1]=pack_probability(e[4],e[5],e[6],e[7]);
                }
                *reinterpret_cast<uint4*>(s.p+Layout{}(cute::make_coord(col,c0)))=*reinterpret_cast<uint4*>(packed);
                *reinterpret_cast<uint4*>(s.p+Layout{}(cute::make_coord(col,c0+16)))=*reinterpret_cast<uint4*>(packed+4);
                if(c0==32) {
                    cutlass::arch::fence_view_async_shared();
                    before_sync();wg_sync<1>();
                    if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_half);
                }
            }
            cutlass::arch::fence_view_async_shared();
            float den=((sums[0]+sums[2])+(sums[4]+sums[6]))+((sums[1]+sums[3])+(sums[5]+sums[7]));
            float inv=rcp_native(den>0.f?den:1.f);
            float lp=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
            wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll
            for(int c0=0;c0<128;c0+=32) {
                float values[32];
                load32(t+addr+c0,values);
                cutlass::arch::fence_view_async_tmem_load();
                if(c0==96) { before_sync();wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.math_free); }
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[c0+k],running[c0+k+1],values[k],values[k+1],inv,a,b);
            }
            m=nm;z=z*a+b;
        }
        float inv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll
        for(int c=0;c<128;c+=32) {
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(running[c+k*2],running[c+k*2+1],inv);
            #pragma unroll
            for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(local,c+k*2)))=*reinterpret_cast<uint4*>(packed+k);
        }
        if(query<stop) p.lse[int64_t(query)*64+head*16+local%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
        wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        if(local<96) {
            for(int page=0;page<16;++page) {
                int slot=page%2;
                if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((page/2)-1)%2);
                int block=s.blocks[page],physical=s.physical[page];
                bool valid=physical>=0;
                int safe_page=valid?physical:0;
                for(int i=local;i<1024;i+=96) {
                    int token=i/8,d=i%8*16;bool present=valid&&block*128+token<length;
                    int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                    Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d))),present);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.kv[slot+2]+VLayout{}(cute::make_coord(d,token))),present);
                }
                cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_view_async_shared();loader_sync();
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
            }
        } else {
            uint64_t dq=descriptor(s.q),dp=descriptor(s.p);
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            for(int page=0;page<16;++page) {
                int slot=page%2;
                if(page>0) cutlass::arch::ClusterBarrier::wait(&s.math_free,(page-1)%2);
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(page/2)%2);after_sync();
                uint64_t dk=descriptor(s.kv[slot]);
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready);
                cutlass::arch::ClusterBarrier::wait(&s.p_half,page%2);after_sync();
                uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[slot+2])),VLayout{}));
                #pragma unroll
                for(int k=0;k<64;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,k!=0,inst_pv);
                cutlass::arch::ClusterBarrier::wait(&s.p_ready,page%2);after_sync();
                #pragma unroll
                for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,true,inst_pv);
                cutlass::arch::umma_arrive(&s.pv_ready);cutlass::arch::umma_arrive(&s.kv_free[slot]);
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    for(int i=tid;i<2048;i+=256) {
        int row=i/16,d=i%16*8,q=qbase+row/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+row%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,256);
}
} // namespace rtp_llm
