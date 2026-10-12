#pragma once

namespace rtp_llm {
struct alignas(128) DualQueryShared {
    alignas(128) uint8_t q[2][16384],p[2][16384],kv[4][16384];
    float inv[2][128],lp[2][128];
    alignas(8) uint64_t kv_ready[2],kv_free[2],s_ready[2],s_free[2],p_half[2],p_ready[2],stats_free[2],o_ready[2],o_free[2];
    uint64_t done;
    uint32_t tmem;
};
static_assert(sizeof(DualQueryShared)<=232448,"dual query exceeds SMEM budget");

template<int Qid>
__device__ __forceinline__ void dual_query_softmax(Params p,DualQueryShared& s,uint32_t t,int batch,int head,int qbase,int start,int stop,int length) {
    cutlass::arch::warpgroup_reg_alloc<168>();
    int local=threadIdx.x%128,warp=local/32;
    uint32_t row_addr=uint32_t(warp*32)<<16;
    int query=qbase+Qid*8+local/16,limit=query-start+length-(stop-start);
    for(int page=0;page<16;++page) {
        cutlass::arch::ClusterBarrier::wait(&s.s_ready[Qid],page%2);after_sync();
        int block=p.topk[head*p.th+int64_t(qbase)*p.tq+page];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[batch*p.tb+block]:-1;
        valid=valid&&physical>=0&&physical<p.pages&&query<stop;
        float inv,lp;
        page_softmax<true>(t+Qid*128,row_addr,block,limit,length,valid,p.alpha,inv,lp,&s.p_half[Qid],s.p[Qid]);
        if(page>0) cutlass::arch::ClusterBarrier::wait(&s.stats_free[Qid],(page-1)%2);
        s.inv[Qid][local]=inv;s.lp[Qid][local]=lp;
        before_sync();wg_sync<Qid+1>();
        if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready[Qid]);
    }
}

__global__ __launch_bounds__(512,1) void grouped_cutlass_dual_query(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<DualQueryShared*>(smem);
    int tid=threadIdx.x,wg=tid/128,local=tid%128,warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*16;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.s_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.s_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_half[i],1);cutlass::arch::ClusterBarrier::init(&s.p_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.stats_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.o_ready[i],1);cutlass::arch::ClusterBarrier::init(&s.o_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.done,1);
    }
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    for(int i=tid;i<8192;i+=512) {
        int qid=i/4096,r=(i%4096)/32,d=i%32*4,q=qbase+qid*8+r/16;
        uint32_t x=q<stop?*reinterpret_cast<const uint32_t*>(p.q+int64_t(q)*p.qn+(head*16+r%16)*p.qh+d):0;
        *reinterpret_cast<uint32_t*>(s.q[qid]+Layout{}(cute::make_coord(r,d)))=x;
    }
    cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if(wg==0) dual_query_softmax<0>(p,s,t,batch,head,qbase,start,stop,length);
    else if(wg==1) dual_query_softmax<1>(p,s,t,batch,head,qbase,start,stop,length);
    else if(wg==2) {
        cutlass::arch::warpgroup_reg_dealloc<112>();
        float m0=-CUDART_INF_F,m1=-CUDART_INF_F,z0=0.f,z1=0.f;
        for(int job=0;job<32;++job) {
            int qid=job%2,page=job/2;
            cutlass::arch::ClusterBarrier::wait(&s.o_ready[qid],page%2);after_sync();
            float inv=s.inv[qid][local],lp=s.lp[qid][local];
            wg_sync<3>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.stats_free[qid]);
            float m=qid==0?m0:m1,z=qid==0?z0:z1,nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll
            for(int c=0;c<128;c+=32) {
                float values[32],running[32]={};load32(t+qid*128+addr+c,values);
                cutlass::arch::fence_view_async_tmem_load();
                if(page>0) {
                    load32(t+256+qid*128+addr+c,running);
                    cutlass::arch::fence_view_async_tmem_load();
                }
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],inv,a,b);
                store32(t+256+qid*128+addr+c,running);
            }
            if(qid==0) {m0=nm;z0=z*a+b;} else {m1=nm;z1=z*a+b;}
            cutlass::arch::fence_view_async_tmem_store();
            before_sync();wg_sync<3>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.s_free[qid]);
        }
        #pragma unroll 1
        for(int qid=0;qid<2;++qid) {
            float mass=qid==0?z0:z1,maximum=qid==0?m0:m1,inv=rcp_native(mass>0.f?mass:1.f);
            auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv)+qid*16384;
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float running[32];
                load32(t+256+qid*128+addr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                #pragma unroll
                for(int k=0;k<32;k+=2) *reinterpret_cast<uint32_t*>(output+OutputLayout{}(cute::make_coord(local,c+k)))=normalize_bf16_pair(running[k],running[k+1],inv);
            }
            int query=qbase+qid*8+local/16;
            if(query<stop) p.lse[int64_t(query)*64+head*16+local%16]=mass>0.f?(maximum+log_native(mass))*0.6931471805599453f:-CUDART_INF_F;
        }
        wg_sync<3>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        if(local<96) {
            using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
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
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            for(int page=0;page<16;++page) {
                int slot=page%2;
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(page/2)%2);
                #pragma unroll
                for(int qid=0;qid<2;++qid) {
                    if(page>0) cutlass::arch::ClusterBarrier::wait(&s.s_free[qid],(page-1)%2);
                    after_sync();
                    uint64_t dq=descriptor(s.q[qid]),dk=descriptor(s.kv[slot]);
                    #pragma unroll
                    for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t+qid*128,k!=0,inst);
                    cutlass::arch::umma_arrive(&s.s_ready[qid]);
                }
                #pragma unroll
                for(int qid=0;qid<2;++qid) {
                    cutlass::arch::ClusterBarrier::wait(&s.p_half[qid],page%2);
                    after_sync();
                    uint64_t dp=descriptor(s.p[qid]);
                    uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[slot+2])),VLayout{}));
                    #pragma unroll
                    for(int k=0;k<64;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+qid*128,k!=0,inst_pv);
                    cutlass::arch::ClusterBarrier::wait(&s.p_ready[qid],page%2);after_sync();
                    #pragma unroll
                    for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+qid*128,true,inst_pv);
                    cutlass::arch::umma_arrive(&s.o_ready[qid]);
                    if(qid==1) cutlass::arch::umma_arrive(&s.kv_free[slot]);
                }
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    for(int i=tid;i<4096;i+=512) {
        int qid=i/2048,r=(i%2048)/16,d=i%16*8,q=qbase+qid*8+r/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+r%16)*128+d)=*reinterpret_cast<uint4*>(output+qid*16384+OutputLayout{}(cute::make_coord(r,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,512);
}
} // namespace rtp_llm
