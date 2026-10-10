#pragma once
#ifndef MSA_Q_TMEM
#define MSA_Q_TMEM 0
#endif
#ifndef MSA_FORCE_ONE_CTA
#define MSA_FORCE_ONE_CTA 0
#endif

namespace rtp_llm {
__global__ __launch_bounds__(256,2) void grouped_cutlass_resident12g2(Params p, const uint32_t* flags) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident12Shared*>(smem);
    int tid=threadIdx.x,local=tid%128,warp=local/32;
    const int group=blockIdx.z;
    const int PP=16/gridDim.z;  // pages per group
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    const int64_t cta7=int64_t(blockIdx.y)*gridDim.x+blockIdx.x;
    const int64_t ncta7=int64_t(gridDim.x)*gridDim.y;
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);cutlass::arch::ClusterBarrier::init(&s.pv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.math_free,1);
        cutlass::arch::ClusterBarrier::init(&s.p_half,1);cutlass::arch::ClusterBarrier::init(&s.p_half2,1);cutlass::arch::ClusterBarrier::init(&s.p_ready,1);
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
        uint64_t* dbg=MSA_TRACE&&p.debug?p.debug+cta7*16*12:nullptr;
        for(int page=group*PP;page<(group+1)*PP;++page) {
            const int li=page-group*PP;
            if(dbg&&local==0) dbg[page*12+0]=clock64();
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,li%2);after_sync();
            if(dbg&&local==0) dbg[page*12+1]=clock64();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0&&query<stop;
            float inv,lp;
            page_softmax<true>(t,addr,block,limit,length,valid,p.alpha,inv,lp,dbg?dbg+page*12:nullptr,&s.p_half,s.p,&s.p_half2);
            wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,li%2);after_sync();
            if(dbg&&local==0) dbg[page*12+10]=clock64();
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            float pre_run[32]={};
            bool has_run=page>0;
            if(has_run) { load32(t+128+addr,pre_run); }
            #pragma unroll 2
            for(int c=0;c<128;c+=32) {
                float values[32],running[32];
                load32(t+addr+c,values);
                if(c==0) {
                    cutlass::arch::fence_view_async_tmem_load();
                    if(has_run) {
                        #pragma unroll
                        for(int k=0;k<32;++k) running[k]=pre_run[k];
                    } else {
                        #pragma unroll
                        for(int k=0;k<32;++k) running[k]=0.f;
                    }
                } else {
                    load32(t+128+addr+c,running);
                    cutlass::arch::fence_view_async_tmem_load();
                }
                if(c==96) { before_sync();wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.math_free); }
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],inv,a,b);
                store32(t+128+addr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();m=nm;z=z*a+b;
            if(dbg&&local==0) dbg[page*12+11]=clock64();
        }
        float inv=rcp_native(z>0.f?z:1.f);
        #pragma unroll 1
        for(int c=0;c<128;c+=32) {
            float running[32];load32(t+128+addr+c,running);cutlass::arch::fence_view_async_tmem_load();
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(running[k*2],running[k*2+1],inv);
            if(query<stop) {
                #pragma unroll
                for(int k=0;k<16;k+=4) {
                    const int64_t base=((int64_t(query)*4+head)*16+group)*16+local%16;
                    *reinterpret_cast<uint4*>(p.partial+base*128+c+k*2)=*reinterpret_cast<uint4*>(packed+k);
                }
            }
        }
        if(query<stop) {
            p.partial_lse[((int64_t(query)*4+head)*16+group)*16+local%16]=
                z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
            if(group==0&&local%16==0) p.counts[int64_t(query)*4+head]=int(gridDim.z);
        }
        wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        if(local<96) {
            grid_dep_wait();  // orders the epoch read; k8/v8 gate on stamps
            const uint32_t* stamps=flags?flags+1:nullptr;
            unsigned e=0;
            if(stamps) e=__ldcg(flags);
            uint64_t* dbg=MSA_TRACE&&p.debug?p.debug+(cta7+ncta7)*16*12:nullptr;
            for(int page=group*PP;page<(group+1)*PP;++page) {
                const int li=page-group*PP;
                if(dbg&&local==0) dbg[page*12+0]=clock64();
                int slot=li%2;
                if(li>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((li/2)-1)%2);
                if(dbg&&local==0) dbg[page*12+1]=clock64();
                int block=s.blocks[page],physical=s.physical[page];
                bool valid=physical>=0;
                if(stamps&&valid) {
                    const volatile uint32_t* st=stamps+int64_t(physical)*4+head;
                    while(*st!=e) __nanosleep(64);
                }
                int safe_page=valid?physical:0;
                for(int i=local;i<1024;i+=96) {
                    int token=i/8,d=i%8*16;bool present=valid&&block*128+token<length;
                    int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                    Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d))),present);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.kv[slot+2]+VLayout{}(cute::make_coord(d,token))),present);
                }
                cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_view_async_shared();loader_sync();
                if(dbg&&local==0) dbg[page*12+2]=clock64();
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
                if(dbg&&local==0) dbg[page*12+3]=clock64();
            }
        } else {
            uint64_t dq=descriptor(s.q),dp=descriptor(s.p);
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            uint64_t* dbg=MSA_TRACE&&p.debug?p.debug+(cta7+2*ncta7)*16*12:nullptr;
            for(int page=group*PP;page<(group+1)*PP;++page) {
                const int li=page-group*PP;
                int slot=li%2;
                if(dbg&&local==96) dbg[page*12+0]=clock64();
                if(li>0) cutlass::arch::ClusterBarrier::wait(&s.math_free,(li-1)%2);
                if(dbg&&local==96) dbg[page*12+1]=clock64();
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(li/2)%2);after_sync();
                if(dbg&&local==96) dbg[page*12+2]=clock64();
                uint64_t dk=descriptor(s.kv[slot]);
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready);
                if(dbg&&local==96) dbg[page*12+3]=clock64();
                cutlass::arch::ClusterBarrier::wait(&s.p_half,li%2);after_sync();
                if(dbg&&local==96) dbg[page*12+4]=clock64();
                uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[slot+2])),VLayout{}));
                QKMma::fma(dp,dv+VLayout{}(cute::make_coord(0,0))/16,t,false,inst_pv);
                cutlass::arch::ClusterBarrier::wait(&s.p_half2,page%2);after_sync();
                QKMma::fma(dp+32/16,dv+VLayout{}(cute::make_coord(0,32))/16,t,true,inst_pv);
                if(dbg&&local==96) dbg[page*12+5]=clock64();
                cutlass::arch::ClusterBarrier::wait(&s.p_ready,li%2);after_sync();
                if(dbg&&local==96) dbg[page*12+6]=clock64();
                #pragma unroll
                for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,true,inst_pv);
                cutlass::arch::umma_arrive(&s.pv_ready);cutlass::arch::umma_arrive(&s.kv_free[slot]);
                if(dbg&&local==96) dbg[page*12+7]=clock64();
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,MSA_Q_TMEM?512:256);
}
} // namespace rtp_llm
