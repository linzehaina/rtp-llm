#pragma once

#ifndef MSA_R4C
#define MSA_R4C 0
#endif

namespace rtp_llm {
struct alignas(128) Resident2Shared {
    alignas(128) uint8_t q[16384],p[16384],kv[4][16384];
    int blocks[16],physical[16];
    alignas(8) uint64_t kv_ready[2],kv_free[2],score_ready[2],pv_ready[2],math_free,p_half,p_ready,done;
    uint32_t tmem;
};
static_assert(sizeof(Resident2Shared)<=232448,"resident2 kernel exceeds SMEM budget");

__global__ __launch_bounds__(256,2) void grouped_cutlass_resident2(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident2Shared*>(smem);
    int tid=threadIdx.x,local=tid%128,warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.score_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.pv_ready[i],1);
        }
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
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
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
        float m=-CUDART_INF_F,z=0.f;
        for(int page=0;page<16;++page) {
            int slot=page%2;
            cutlass::arch::ClusterBarrier::wait(&s.score_ready[slot],(page/2)%2);after_sync();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0&&query<stop;
            float inv,lp;
            page_softmax<true>(t+slot*128,addr,block,limit,length,valid,p.alpha,inv,lp,nullptr,&s.p_half,s.p);
            wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready[slot],(page/2)%2);after_sync();
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float values[32],running[32];
                load32(t+slot*128+addr+c,values);
                load32(t+256+addr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],inv,a,b);
                store32(t+256+addr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();m=nm;z=z*a+b;
            before_sync();wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.math_free);
        }
        float inv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll 1
        for(int c=0;c<128;c+=32) {
            float running[32];load32(t+256+addr+c,running);cutlass::arch::fence_view_async_tmem_load();
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
#if MSA_R4C
                if(page>=1) cutlass::arch::ClusterBarrier::wait(&s.math_free,(page-1)%2);
#else
                if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.math_free,page%2);
#endif
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(page/2)%2);after_sync();
                uint64_t dk=descriptor(s.kv[slot]);
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t+slot*128,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready[slot]);
                cutlass::arch::ClusterBarrier::wait(&s.p_half,page%2);after_sync();
                uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[slot+2])),VLayout{}));
                #pragma unroll
                for(int k=0;k<64;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+slot*128,k!=0,inst_pv);
                cutlass::arch::ClusterBarrier::wait(&s.p_ready,page%2);after_sync();
                #pragma unroll
                for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+slot*128,true,inst_pv);
                cutlass::arch::umma_arrive(&s.pv_ready[slot]);cutlass::arch::umma_arrive(&s.kv_free[slot]);
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    for(int i=tid;i<2048;i+=256) {
        int row=i/16,d=i%16*8,q=qbase+row/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+row%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,512);
}
} // namespace rtp_llm
