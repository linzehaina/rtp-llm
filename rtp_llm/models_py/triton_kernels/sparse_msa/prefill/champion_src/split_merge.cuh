#pragma once

namespace rtp_llm {
struct alignas(128) SplitMergeShared {
    alignas(128) uint8_t q[16384],kv[4][16384];
    float inv[2][128],lp[2][128];
    alignas(8) uint64_t kv_ready[2],kv_free[2],s_ready[2],s_free[2],p_half[2],p_ready[2],stats_free[2];
    uint64_t o_ready,o_free,done;
    uint32_t tmem;
};

__global__ __launch_bounds__(512,1) void grouped_cutlass_split_merge(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<SplitMergeShared*>(smem);
    int tid=threadIdx.x,wg=tid/128,local=tid%128,warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.s_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.s_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_half[i],1);cutlass::arch::ClusterBarrier::init(&s.p_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.stats_free[i],2);
        }
        cutlass::arch::ClusterBarrier::init(&s.o_ready,1);cutlass::arch::ClusterBarrier::init(&s.o_free,2);
        cutlass::arch::ClusterBarrier::init(&s.done,2);
    }
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    for(int i=tid;i<1024;i+=512) {
        int row=i/8,d=i%8*16,q=qbase+row/16,safe_q=q<stop?q:start;
        Copy::copy(*reinterpret_cast<const uint4*>(p.q+int64_t(safe_q)*p.qn+(head*16+row%16)*p.qh+d),
                   *reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(row,d))),q<stop);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();
    cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if(wg==0) {
        cutlass::arch::warpgroup_reg_alloc<192>();
        int query=qbase+local/16,limit=query-start+length-(stop-start);
        for(int page=0;page<16;++page) {
            int slot=page%2;
            cutlass::arch::ClusterBarrier::wait(&s.s_ready[slot],(page/2)%2);after_sync();
            int block=p.topk[head*p.th+int64_t(qbase)*p.tq+page];
            bool valid=block>=0&&block<p.max_pages&&block*128<length;
            int physical=valid?p.table[batch*p.tb+block]:-1;
            valid=valid&&physical>=0&&physical<p.pages&&query<stop;
            float inv,lp;
            page_softmax(t+slot*128,addr,block,limit,length,valid,p.alpha,inv,lp,nullptr,&s.p_half[slot]);
            if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.stats_free[slot],((page/2)-1)%2);
            s.inv[slot][local]=inv;s.lp[slot][local]=lp;
            before_sync();wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready[slot]);
        }
    } else if(wg<3) {
        cutlass::arch::warpgroup_reg_alloc<128>();
        int base=(wg-1)*64;
        float m=-CUDART_INF_F,z=0.f;
        for(int page=0;page<16;++page) {
            int slot=page%2;
            cutlass::arch::ClusterBarrier::wait(&s.o_ready,page%2);after_sync();
            float inv=s.inv[slot][local],lp=s.lp[slot][local];
            if(wg==1) wg_sync<2>(); else wg_sync<3>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.stats_free[slot]);
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll
            for(int c=0;c<64;c+=32) {
                float values[32],running[32]={};load32(t+256+addr+base+c,values);
                if(page>0) load32(t+384+addr+base+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],inv,a,b);
                store32(t+384+addr+base+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();m=nm;z=z*a+b;
            before_sync();if(wg==1) wg_sync<2>(); else wg_sync<3>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.o_free);
        }
        float inv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll 1
        for(int c=0;c<64;c+=32) {
            float running[32];load32(t+384+addr+base+c,running);cutlass::arch::fence_view_async_tmem_load();
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(running[k*2],running[k*2+1],inv);
            #pragma unroll
            for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(local,base+c+k*2)))=*reinterpret_cast<uint4*>(packed+k);
        }
        int query=qbase+local/16;
        if(wg==1&&query<stop) p.lse[int64_t(query)*64+head*16+local%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
        if(wg==1) wg_sync<2>(); else wg_sync<3>();
        if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        if(local<96) {
            for(int page=0;page<16;++page) {
                int slot=page%2;
                if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((page/2)-1)%2);
                int block=p.topk[head*p.th+int64_t(qbase)*p.tq+page];
                bool valid=block>=0&&block<p.max_pages&&block*128<length;
                int physical=valid?p.table[batch*p.tb+block]:-1;
                valid=valid&&physical>=0&&physical<p.pages;
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
            uint64_t dq=descriptor(s.q);
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            for(int job=0;job<17;++job) {
                if(job<16) {
                    int slot=job%2;
                    if(job>=2) cutlass::arch::ClusterBarrier::wait(&s.s_free[slot],((job/2)-1)%2);
                    cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(job/2)%2);after_sync();
                    uint64_t dk=descriptor(s.kv[slot]);
                    #pragma unroll
                    for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t+slot*128,k!=0,inst);
                    cutlass::arch::umma_arrive(&s.s_ready[slot]);
                }
                if(job>0) {
                    int page=job-1,slot=page%2;
                    cutlass::arch::ClusterBarrier::wait(&s.p_half[slot],(page/2)%2);
                    if(page>0) cutlass::arch::ClusterBarrier::wait(&s.o_free,(page-1)%2);
                    after_sync();
                    uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[slot+2])),VLayout{}));
                    #pragma unroll
                    for(int k=0;k<64;k+=32) PVMmaMN::fma(t+slot*128+64+k/4,dv+VLayout{}(cute::make_coord(0,k))/16,t+256,k!=0,inst_pv);
                    cutlass::arch::ClusterBarrier::wait(&s.p_ready[slot],(page/2)%2);after_sync();
                    #pragma unroll
                    for(int k=64;k<128;k+=32) PVMmaMN::fma(t+slot*128+64+k/4,dv+VLayout{}(cute::make_coord(0,k))/16,t+256,true,inst_pv);
                    cutlass::arch::umma_arrive(&s.o_ready);cutlass::arch::umma_arrive(&s.s_free[slot]);cutlass::arch::umma_arrive(&s.kv_free[slot]);
                }
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    for(int i=tid;i<2048;i+=512) {
        int r=i/16,d=i%16*8,q=qbase+r/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+r%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(r,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,512);
}
} // namespace rtp_llm
