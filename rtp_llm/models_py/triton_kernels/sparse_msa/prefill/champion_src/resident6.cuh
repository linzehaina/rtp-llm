#pragma once

namespace rtp_llm {
struct alignas(128) Resident6Shared {
    alignas(128) uint8_t q[16384],p[16384],kv[4][16384];
    int blocks[16],physical[16];
    float halfmax[2][128];
    float denchain[128][8];
    float stats[128][2];
    alignas(8) uint64_t kv_ready[2],kv_free[2],score_ready,pv_ready,math_free;
    uint32_t tmem;
};
static_assert(sizeof(Resident6Shared)<=232448,"resident6 kernel exceeds SMEM budget");

__device__ __forceinline__ void zero_regs6(float* r) {
    #pragma unroll
    for(int i=0;i<32;++i) r[i]=0.f;
}

__device__ __forceinline__ void kv_load6(Params p,Resident6Shared& s,int page,int head,int length,int tid) {
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    int slot=page%2;
    if(page>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((page-2)/2)%2);
    int block=s.blocks[page],physical=s.physical[page];
    bool valid=physical>=0;
    int safe_page=valid?physical:0;
    for(int i=tid;i<1024;i+=256) {
        int token=i/8,d=i%8*16;bool present=valid&&block*128+token<length;
        int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
        Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d))),present);
        Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.kv[2+slot]+VLayout{}(cute::make_coord(d,token))),present);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
}

__global__ __launch_bounds__(256,2) void grouped_cutlass_resident6(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident6Shared*>(smem);
    int tid=threadIdx.x,half=tid/128,local=tid%128,warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        cutlass::arch::ClusterBarrier::init(&s.kv_ready[0],256);cutlass::arch::ClusterBarrier::init(&s.kv_ready[1],256);
        cutlass::arch::ClusterBarrier::init(&s.kv_free[0],1);cutlass::arch::ClusterBarrier::init(&s.kv_free[1],1);
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);cutlass::arch::ClusterBarrier::init(&s.pv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.math_free,1);
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
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    for(int i=tid;i<1024;i+=256) {
        int row=i/8,d=i%8*16,q=qbase+row/16,safe_q=q<stop?q:start;
        Copy::copy(*reinterpret_cast<const uint4*>(p.q+int64_t(safe_q)*p.qn+(head*16+row%16)*p.qh+d),
                   *reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(row,d))),q<stop);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();
    cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();
    __syncthreads();
    kv_load6(p,s,0,head,length,tid);
    kv_load6(p,s,1,head,length,tid);
    if(tid==0) {
        uint64_t dq=descriptor(s.q);
        uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        cutlass::arch::ClusterBarrier::wait(&s.kv_ready[0],0);after_sync();
        uint64_t dk=descriptor(s.kv[0]);
        #pragma unroll
        for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,s.tmem,k!=0,inst);
        cutlass::arch::umma_arrive(&s.score_ready);
    }
    __syncthreads();
    uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    int query=qbase+local/16,limit=query-start+length-(stop-start);
    float m=-CUDART_INF_F,z=0.f;
    uint64_t dq=descriptor(s.q),dp=descriptor(s.p);
    uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
    uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
    for(int page=0;page<16;++page) {
        cutlass::arch::ClusterBarrier::wait(&s.score_ready,page%2);after_sync();
        int block=s.blocks[page];
        bool valid=s.physical[page]>=0&&query<stop;
        int last=valid?min(length-1,limit)-block*128:-1;
        float v[64],tile_max[2];
        #pragma unroll
        for(int c=0;c<64;c+=32) load32max(t+half*64+addr+c,v+c,tile_max[c/32]);
        cutlass::arch::fence_view_async_tmem_load();
        float partial;
        int fast_last=half?127:63;
        if(__all_sync(0xffffffff,last>=fast_last)) {
            partial=max_native(tile_max[0],tile_max[1]);
        } else {
            float maxima[8];
            #pragma unroll
            for(int i=0;i<8;++i) maxima[i]=-CUDART_INF_F;
            #pragma unroll
            for(int c=0;c<64;++c) {
                float masked;
                asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                    : "=f"(masked) : "f"(v[c]),"f"(-CUDART_INF_F),"r"(half*64+c),"r"(last));
                v[c]=masked;
                maxima[c%8]=max_native(maxima[c%8],v[c]);
            }
            partial=max_native(max_native(max_native(maxima[0],maxima[1]),max_native(maxima[2],maxima[3])),
                               max_native(max_native(maxima[4],maxima[5]),max_native(maxima[6],maxima[7])));
        }
        s.halfmax[half][local]=partial;
        cutlass::arch::fence_view_async_shared();
        asm volatile("bar.sync 1, 256;" ::: "memory");
        float maximum=max_native(s.halfmax[0][local],s.halfmax[1][local]);
        float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
        #pragma unroll
        for(int c=0;c<64;c+=2) {
            affine_pair(v[c],v[c+1],p.alpha,shift);
            int gc=half*64+c;
            v[c]=(gc>=32&&gc<96&&gc%16>=12)?exp_emulated(v[c]):exp_native(v[c]);
            v[c+1]=(gc+1>=32&&gc+1<96&&(gc+1)%16>=12)?exp_emulated(v[c+1]):exp_native(v[c+1]);
        }
        uint32_t packed[16];
        #pragma unroll
        for(int c=0;c<16;++c) packed[c]=pack_probability(v[c*4],v[c*4+1],v[c*4+2],v[c*4+3]);
        #pragma unroll
        for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(s.p+Layout{}(cute::make_coord(local,half*64+c*4)))=*reinterpret_cast<uint4*>(packed+c);
        if(half==0) {
            float sums[8]={};
            #pragma unroll
            for(int j=0;j<8;++j) {
                #pragma unroll
                for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[j*8+c],v[j*8+c+1]);
            }
            #pragma unroll
            for(int c=0;c<8;++c) s.denchain[local][c]=sums[c];
            cutlass::arch::fence_view_async_shared();
            asm volatile("bar.sync 2, 128;" ::: "memory");
            if(tid==0) {
                uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[2+page%2])),VLayout{}));
                #pragma unroll
                for(int k=0;k<64;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,k!=0,inst_pv);
            }
        }
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if(tid==0) {
            uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[2+page%2])),VLayout{}));
            #pragma unroll
            for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,true,inst_pv);
            cutlass::arch::umma_arrive(&s.pv_ready);cutlass::arch::umma_arrive(&s.kv_free[page%2]);
        }
        float inv,lp;
        if(half==1) {
            float full[8];
            #pragma unroll
            for(int c=0;c<8;++c) full[c]=s.denchain[local][c];
            #pragma unroll
            for(int j=0;j<8;++j) {
                #pragma unroll
                for(int c=0;c<8;c+=2) add_pair(full[c],full[c+1],v[8*j+c],v[8*j+c+1]);
            }
            float den=((full[0]+full[2])+(full[4]+full[6]))+((full[1]+full[3])+(full[5]+full[7]));
            inv=rcp_native(den>0.f?den:1.f);
            lp=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
            s.stats[local][0]=inv;s.stats[local][1]=lp;
            cutlass::arch::fence_view_async_shared();
        }
        asm volatile("bar.sync 1, 256;" ::: "memory");
        inv=s.stats[local][0];lp=s.stats[local][1];
        cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
        float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
        float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
        {
            float v0[32],v1[32],r0[32],r1[32];
            load32(t+half*64+addr,v0);
            if(page>0) load32(t+128+half*64+addr,r0); else zero_regs6(r0);
            cutlass::arch::fence_view_async_tmem_load();
            load32(t+half*64+addr+32,v1);
            if(page>0) load32(t+128+half*64+addr+32,r1); else zero_regs6(r1);
            #pragma unroll
            for(int k=0;k<32;k+=2) merge_pair(r0[k],r0[k+1],v0[k],v0[k+1],inv,a,b);
            store32(t+128+half*64+addr,r0);
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for(int k=0;k<32;k+=2) merge_pair(r1[k],r1[k+1],v1[k],v1[k+1],inv,a,b);
            store32(t+128+half*64+addr+32,r1);
        }
        cutlass::arch::fence_view_async_tmem_store();
        if(page+2<16) kv_load6(p,s,page+2,head,length,tid);
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if(tid==0) {
            cutlass::arch::ClusterBarrier::arrive(&s.math_free);
            if(page+1<16) {
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[(page+1)%2],((page+1)/2)%2);
                uint64_t dk=descriptor(s.kv[(page+1)%2]);
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready);
            }
        }
        m=nm;z=z*a+b;
    }
    float inv=rcp_native(z>0.f?z:1.f);
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    {
        float r0[32],r1[32];
        load32(t+128+half*64+addr,r0);load32(t+128+half*64+addr+32,r1);
        cutlass::arch::fence_view_async_tmem_load();
        uint32_t packed[32];
        #pragma unroll
        for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(r0[k*2],r0[k*2+1],inv);
        #pragma unroll
        for(int k=0;k<16;++k) packed[16+k]=normalize_bf16_pair(r1[k*2],r1[k*2+1],inv);
        #pragma unroll
        for(int k=0;k<32;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(local,half*64+k*2)))=*reinterpret_cast<uint4*>(packed+k);
    }
    if(half==0&&query<stop) p.lse[int64_t(query)*64+head*16+local%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
    before_sync();__syncthreads();
    for(int i=tid;i<2048;i+=256) {
        int row=i/16,d=i%16*8,q=qbase+row/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+row%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,256);
}
} // namespace rtp_llm
