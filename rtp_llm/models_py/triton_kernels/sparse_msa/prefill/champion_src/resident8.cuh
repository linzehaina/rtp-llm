#pragma once
#ifndef MSA_Q_TMEM
#define MSA_Q_TMEM 0
#endif

namespace rtp_llm {
struct alignas(128) Resident8Shared {
    alignas(128) uint8_t q[16384],p[16384],kv[4][16384];
    alignas(128) uint8_t stage_k[2][8192],stage_v[2][8192],stage_ks[2][1024],stage_vs[2][1024];
    int blocks[16],physical[16];
    alignas(8) uint64_t kv_ready[2],kv_free[2],score_ready,pv_ready,pv_half,math_free,p_half,p_ready,done;
    uint32_t tmem;
};
static_assert(sizeof(Resident8Shared)<=232448,"resident8 kernel exceeds SMEM budget");

__global__ __launch_bounds__(256,2) void grouped_cutlass_resident8(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident8Shared*>(smem);
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
        cutlass::arch::ClusterBarrier::init(&s.pv_half,1);
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
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(MSA_Q_TMEM?512:256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    for(int i=tid;i<1024;i+=256) {
        int row=i/8,d=i%8*16,q=qbase+row/16,safe_q=q<stop?q:start;
        Copy::copy(*reinterpret_cast<const uint4*>(p.q+int64_t(safe_q)*p.qn+(head*16+row%16)*p.qh+d),
                   *reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(row,d))),q<stop);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if constexpr(MSA_Q_TMEM) {
        if(tid<128) {
            uint32_t words[32];
            #pragma unroll
            for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(words+c)=*reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(local,c*4)));
            store32(t+256+addr,words);cutlass::arch::fence_view_async_tmem_store();before_sync();
        }
        __syncthreads();
    }
    if(tid<128) {
        cutlass::arch::warpgroup_reg_alloc<192>();
        int query=qbase+local/16,limit=query-start+length-(stop-start);
        float m=-CUDART_INF_F,z=0.f;
        for(int page=0;page<16;++page) {
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,page%2);after_sync();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0&&query<stop;
            float inv,lp;
            page_softmax<true>(t,addr,block,limit,length,valid,p.alpha,inv,lp,&s.p_half,s.p);
            wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float values[32],running[32]={};load32(t+addr+c,values);
                if(page>0) load32(t+128+addr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],inv,a,b);
                store32(t+128+addr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();m=nm;z=z*a+b;
            before_sync();wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.math_free);
        }
        float inv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll 1
        for(int c=0;c<128;c+=32) {
            float running[32];load32(t+128+addr+c,running);cutlass::arch::fence_view_async_tmem_load();
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(running[k*2],running[k*2+1],inv);
            #pragma unroll
            for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(local,c+k*2)))=*reinterpret_cast<uint4*>(packed+k);
        }
        if(query<stop) p.lse[int64_t(query)*64+head*16+local%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
        wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        if(local<96) {
            auto issue_packed=[&](int page,int slot) {
                int block=s.blocks[page],physical=s.physical[page];
                bool valid=physical>=0;
                int safe_page=valid?physical:0;
                for(int i=local;i<512;i+=96) {
                    Copy::copy(*reinterpret_cast<const uint4*>(p.k+int64_t(safe_page)*p.kp+head*8192+i*16),
                               *reinterpret_cast<uint4*>(s.stage_k[slot]+i*16),valid);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.v+int64_t(safe_page)*p.vp+head*8192+i*16),
                               *reinterpret_cast<uint4*>(s.stage_v[slot]+i*16),valid);
                }
                if(local<64) {
                    Copy::copy(*reinterpret_cast<const uint4*>(p.ks+int64_t(safe_page)*4096+head*1024+local*16),
                               *reinterpret_cast<uint4*>(s.stage_ks[slot]+local*16),valid);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.vs+int64_t(safe_page)*4096+head*1024+local*16),
                               *reinterpret_cast<uint4*>(s.stage_vs[slot]+local*16),valid);
                }
            };
            auto decode_staged=[&](int page,int slot) {
                int block=s.blocks[page],physical=s.physical[page];
                bool valid=physical>=0;
                for(int i=local;i<1024;i+=96) {
                    int token=i/8,g=i%8,d=g*16;
                    int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
                    uint64_t pk=*reinterpret_cast<const uint64_t*>(s.stage_k[slot]+token*64+d/2);
                    uint64_t pv=*reinterpret_cast<const uint64_t*>(s.stage_v[slot]+token*64+d/2);
                    uint8_t sk=s.stage_ks[slot][sf],sv=s.stage_vs[slot][sf];
                    uint4 key,value;
                    #pragma unroll
                    for(int c=0;c<4;++c) {
                        reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(pk>>(c*16)),sk);
                        reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(pv>>(c*16)),sv);
                    }
                    *reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d)))=key;
                    *reinterpret_cast<uint4*>(s.kv[2+slot]+VLayout{}(cute::make_coord(d,token)))=value;
                }
            };
            issue_packed(0,0);
            cute::cp_async_fence();
            issue_packed(1,1);
            cute::cp_async_fence();
            for(int page=0;page<16;++page) {
                int slot=page%2;
                if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((page/2)-1)%2);
                cute::cp_async_wait<1>();
                cutlass::arch::fence_view_async_shared();loader_sync();
                decode_staged(page,slot);
                if(page+2<16) { issue_packed(page+2,slot); cute::cp_async_fence(); }
                cutlass::arch::fence_view_async_shared();loader_sync();
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
                for(int k=0;k<128;k+=32) {
                    if constexpr(MSA_Q_TMEM) {
                        using QT=cute::SM100_MMA_F8F6F4_TS<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>;
                        QT::fma(t+256+k/4,dk+k/16,t,k!=0,inst);
                    } else QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                }
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
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,MSA_Q_TMEM?512:256);
}
} // namespace rtp_llm
