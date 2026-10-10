#pragma once

namespace rtp_llm {
struct alignas(128) Resident4Shared {
    alignas(128) uint8_t q[16384],pe[16384],po[16384],kv[8][16384];
    int blocks[16],physical[16];
    float mz[2][128][2];
    alignas(8) uint64_t kv_ready[4],kv_free[4];
    uint64_t score_ready[2],p_half[2],p_ready[2],pv_ready[2],math_free[2],done;
    uint32_t tmem;
};
static_assert(sizeof(Resident4Shared)<=232448,"resident4 kernel exceeds SMEM budget");

__global__ __launch_bounds__(384,1) void grouped_cutlass_resident4(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident4Shared*>(smem);
    int tid=threadIdx.x;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<4;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
        }
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.score_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.p_half[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.pv_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.math_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.done,1);
    }
    if(tid<16) {
        #pragma unroll
        for(int i=0;i<16;++i) {
            int block=p.topk[head*p.th+int64_t(qbase)*p.tq+i];
            bool valid=block>=0&&block<p.max_pages&&block*128<length;
            int physical=valid?p.table[batch*p.tb+block]:-1;
            s.blocks[i]=block;s.physical[i]=(valid&&physical>=0&&physical<p.pages)?physical:-1;
        }
    }
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    for(int i=tid;i<1024;i+=384) {
        int row=i/8,d=i%8*16,q=qbase+row/16,safe_q=q<stop?q:start;
        Copy::copy(*reinterpret_cast<const uint4*>(p.q+int64_t(safe_q)*p.qn+(head*16+row%16)*p.qh+d),
                   *reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(row,d))),q<stop);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    if(tid<256) {
        int chain=tid/128,local=tid%128,warp=local/32;
        cutlass::arch::warpgroup_reg_alloc<192>();
        int query=qbase+local/16,limit=query-start+length-(stop-start);
        uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
        float m=-CUDART_INF_F,z=0.f;
        uint8_t* pbuf[2]={s.pe,s.po};
        for(int q2=0;q2<8;++q2) {
            int page=2*q2+chain;
            cutlass::arch::ClusterBarrier::wait(&s.score_ready[chain],q2%2);after_sync();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0&&query<stop;
            float inv,lp;
            page_softmax<true>(t+chain*128,addr,block,limit,length,valid,p.alpha,inv,lp,nullptr,&s.p_half[chain],pbuf[chain]);
            if(chain==0) wg_sync<1>(); else wg_sync<2>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready[chain]);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready[chain],q2%2);after_sync();
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float values[32],running[32]={};
                load32(t+chain*128+addr+c,values);
                if(page>0) load32(t+256+chain*128+addr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],inv,a,b);
                store32(t+256+chain*128+addr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();
            m=nm;z=z*a+b;
            before_sync();
            if(chain==0) wg_sync<1>(); else wg_sync<2>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.math_free[chain]);
        }
        s.mz[chain][local][0]=m;s.mz[chain][local][1]=z;
        cutlass::arch::fence_view_async_shared();
        before_sync();
    } else if(tid<352) {
        int llocal=tid-256;
        cutlass::arch::warpgroup_reg_dealloc<64>();
        for(int page=0;page<16;++page) {
            int slot=page%4;
            if(page>=4) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((page/4)-1)%2);
            int block=s.blocks[page],physical=s.physical[page];
            bool valid=physical>=0;
            int safe_page=valid?physical:0;
            for(int i=llocal;i<1024;i+=96) {
                int token=i/8,d=i%8*16;bool present=valid&&block*128+token<length;
                int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d))),present);
                Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.kv[4+slot]+VLayout{}(cute::make_coord(d,token))),present);
            }
            cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_view_async_shared();loader_sync();
            if(llocal==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        uint64_t dq=descriptor(s.q);
        uint64_t dp[2]={descriptor(s.pe),descriptor(s.po)};
        uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
        uint32_t t=s.tmem;
        for(int page=0;page<16;++page) {
            int chain=page%2,cp=page/2;
            if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.math_free[chain],((page-2)/2)%2);
            cutlass::arch::ClusterBarrier::wait(&s.kv_ready[page%4],(page/4)%2);after_sync();
            uint64_t dk=descriptor(s.kv[page%4]);
            #pragma unroll
            for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t+chain*128,k!=0,inst);
            cutlass::arch::umma_arrive(&s.score_ready[chain]);
            cutlass::arch::ClusterBarrier::wait(&s.p_half[chain],cp%2);after_sync();
            uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[4+page%4])),VLayout{}));
            #pragma unroll
            for(int k=0;k<64;k+=32) QKMma::fma(dp[chain]+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+chain*128,k!=0,inst_pv);
            cutlass::arch::ClusterBarrier::wait(&s.p_ready[chain],cp%2);after_sync();
            #pragma unroll
            for(int k=64;k<128;k+=32) QKMma::fma(dp[chain]+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+chain*128,true,inst_pv);
            cutlass::arch::umma_arrive(&s.pv_ready[chain]);cutlass::arch::umma_arrive(&s.kv_free[page%4]);
        }
    }
    __syncthreads();
    if(tid>=128&&tid<256) {
        int local=tid-128,warp=local/32;
        uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
        int query=qbase+local/16;
        float mo=s.mz[1][local][0],zo=s.mz[1][local][1];
        float me=s.mz[0][local][0],ze=s.mz[0][local][1];
        float mm=max_native(mo,me),safe=mm==-CUDART_INF_F?0.f:mm;
        float ae=ze>0.f?exp_native(me-safe):0.f,ao=zo>0.f?exp_native(mo-safe):0.f;
        float zz=ze*ae+zo*ao;
        float inv=rcp_native(zz>0.f?zz:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll 1
        for(int c=0;c<128;c+=32) {
            float rve[32],rvo[32];
            load32(t+256+addr+c,rve);load32(t+384+addr+c,rvo);
            cutlass::arch::fence_view_async_tmem_load();
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) {
                float ce=rve[k*2]*ae+rvo[k*2]*ao,co=rve[k*2+1]*ae+rvo[k*2+1]*ao;
                packed[k]=normalize_bf16_pair(ce,co,inv);
            }
            #pragma unroll
            for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(local,c+k*2)))=*reinterpret_cast<uint4*>(packed+k);
        }
        if(query<stop) p.lse[int64_t(query)*64+head*16+local%16]=zz>0.f?(mm+log_native(zz))*0.6931471805599453f:-CUDART_INF_F;
    }
    before_sync();__syncthreads();
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    for(int i=tid;i<2048;i+=384) {
        int row=i/16,d=i%16*8,q=qbase+row/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+row%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(s.tmem,512);
}
} // namespace rtp_llm
