#pragma once

// resident11: softmax/merge warp split. Math warps 0-3 run softmax only.
// The merge runs on the mostly-idle loader warps 4-6 (rows 0-95, their own
// datapath windows) and on the MMA warp 7 (rows 96-127, its window), so
// merge(page) overlaps softmax(page+1). QK(page+1) is released by the
// merge value-load completion (merge_loads barrier, 4 arriving warps).

namespace rtp_llm {

struct alignas(128) Resident11Shared {
    alignas(128) uint8_t q[16384], p[16384], kv[4][16384];
    int blocks[16], physical[16];
    float inv[2][128], lp[2][128];
    alignas(8) uint64_t kv_ready[2], score_ready, pv_ready, p_half, p_ready, merge_loads;
    uint32_t tmem;
};
static_assert(sizeof(Resident11Shared)<=232448,"resident11 kernel exceeds SMEM budget");

__global__ __launch_bounds__(256,2) void grouped_cutlass_resident11(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Resident11Shared*>(smem);
    int tid=threadIdx.x,local=tid%128,warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);cutlass::arch::ClusterBarrier::init(&s.pv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.p_half,1);cutlass::arch::ClusterBarrier::init(&s.p_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.merge_loads,4);
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
    uint32_t t=s.tmem;
    if(tid<128) {
        // math warps 0-3: softmax only, lockstep.
        cutlass::arch::warpgroup_reg_alloc<176>();
        const uint32_t addr=uint32_t(warp*32)<<16;
        int query=qbase+local/16,limit=query-start+length-(stop-start);
        for(int page=0;page<16;++page) {
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,page%2);after_sync();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0&&query<stop;
            int last=valid?min(length-1,limit)-block*128:-1;
            float v[128],tile_max[4];
            #pragma unroll
            for(int c=0;c<128;c+=32) load32max(t+addr+c,v+c,tile_max[c/32]);
            cutlass::arch::fence_view_async_tmem_load();
            float maximum;
            if(__all_sync(0xffffffff,last>=127)) {
                maximum=max_native(max_native(tile_max[0],tile_max[1]),max_native(tile_max[2],tile_max[3]));
            } else {
                float maxima[8];
                #pragma unroll
                for(int i=0;i<8;++i) maxima[i]=-CUDART_INF_F;
                #pragma unroll
                for(int c=0;c<128;++c) {
                    float masked;
                    asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                        : "=f"(masked) : "f"(v[c]),"f"(-CUDART_INF_F),"r"(c),"r"(last));
                    v[c]=masked;
                    maxima[c%8]=max_native(maxima[c%8],v[c]);
                }
                maximum=max_native(max_native(max_native(maxima[0],maxima[1]),max_native(maxima[2],maxima[3])),
                                   max_native(max_native(maxima[4],maxima[5]),max_native(maxima[6],maxima[7])));
            }
            float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
            #pragma unroll
            for(int c=0;c<64;c+=2) {
                affine_pair(v[c],v[c+1],p.alpha,shift);
                v[c]=exp_native(v[c]);
                v[c+1]=exp_native(v[c+1]);
            }
            uint32_t packed_first[16];
            #pragma unroll
            for(int c=0;c<16;++c) packed_first[c]=pack_probability(v[c*4],v[c*4+1],v[c*4+2],v[c*4+3]);
            #pragma unroll
            for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(s.p+Layout{}(cute::make_coord(local,c*4)))=*reinterpret_cast<uint4*>(packed_first+c);
            cutlass::arch::fence_view_async_shared();
            before_sync();wg_sync<1>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_half);
            #pragma unroll
            for(int c=64;c<128;c+=2) {
                affine_pair(v[c],v[c+1],p.alpha,shift);
                v[c]=exp_native(v[c]);
                v[c+1]=exp_native(v[c+1]);
            }
            float sums[8]={};
            #pragma unroll
            for(int j=0;j<16;++j) {
                #pragma unroll
                for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[j*8+c],v[j*8+c+1]);
            }
            float den=((sums[0]+sums[2])+(sums[4]+sums[6]))+((sums[1]+sums[3])+(sums[5]+sums[7]));
            float inv=rcp_native(den>0.f?den:1.f);
            float lp=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
            s.inv[page%2][local]=inv;
            s.lp[page%2][local]=lp;
            uint32_t packed[16];
            #pragma unroll
            for(int c=0;c<16;++c) packed[c]=pack_probability(v[64+c*4],v[64+c*4+1],v[64+c*4+2],v[64+c*4+3]);
            #pragma unroll
            for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(s.p+Layout{}(cute::make_coord(local,64+c*4)))=*reinterpret_cast<uint4*>(packed+c);
            cutlass::arch::fence_view_async_shared();
            before_sync();wg_sync<1>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
        }
    } else if(local<96) {
        // loader warps 4-6: KV prefetch + merge rows 0-95.
        cutlass::arch::warpgroup_reg_dealloc<80>();
        const int lw=(tid-128)/32;
        const uint32_t maddr=uint32_t(lw*32)<<16;
        const int row=local;
        float m=-CUDART_INF_F,z=0.f;
        auto kv_load=[&](int page,int slot) {
            int block=s.blocks[page],physical=s.physical[page];
            bool valid=physical>=0;
            int safe_page=valid?physical:0;
            for(int i=local;i<1024;i+=96) {
                int token=i/8,d=i%8*16;bool present=valid&&block*128+token<length;
                int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d))),present);
                Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.kv[slot+2]+VLayout{}(cute::make_coord(d,token))),present);
            }
        };
        kv_load(0,0);
        cute::cp_async_fence();
        kv_load(1,1);
        cute::cp_async_fence();cute::cp_async_wait<0>();
        cutlass::arch::fence_view_async_shared();
        loader_sync();
        if(local==0) { cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[0]); cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[1]); }
        for(int page=0;page<16;++page) {
            int slot=page%2;
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
            cutlass::arch::ClusterBarrier::wait(&s.p_ready,page%2);
            float minv=s.inv[page%2][row],mlp=s.lp[page%2][row];
            if(page+2<16) { kv_load(page+2,slot); cute::cp_async_fence(); }
            float nm=max_native(m,mlp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=mlp!=-CUDART_INF_F?exp_native(mlp-safe):0.f;
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float values[32],running[32];
                load32(t+maddr+c,values);
                load32(t+128+maddr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                if(c==96&&local%32==0) cutlass::arch::ClusterBarrier::arrive(&s.merge_loads);
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],minv,a,b);
                store32(t+128+maddr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();
            m=nm;z=z*a+b;
            if(page+2<16) { cute::cp_async_wait<0>();cutlass::arch::fence_view_async_shared(); }
            loader_sync();
            if(local==0&&page+2<16) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
        }
        float finv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll 1
        for(int c=0;c<128;c+=32) {
            float running[32];load32(t+128+maddr+c,running);cutlass::arch::fence_view_async_tmem_load();
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(running[k*2],running[k*2+1],finv);
            #pragma unroll
            for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,c+k*2)))=*reinterpret_cast<uint4*>(packed+k);
        }
        int q=qbase+row/16;
        if(q<stop) p.lse[int64_t(q)*64+head*16+row%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
    } else {
        // warp 7: MMA issue + merge rows 96-127 (its datapath window).
        cutlass::arch::warpgroup_reg_dealloc<80>();
        const uint32_t maddr=uint32_t(96)<<16;
        const int row=local;
        uint64_t dq=descriptor(s.q),dp=descriptor(s.p);
        uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
        float m=-CUDART_INF_F,z=0.f;
        cutlass::arch::ClusterBarrier::wait(&s.kv_ready[0],0);after_sync();
        {
            uint64_t dk=descriptor(s.kv[0]);
            #pragma unroll
            for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
            cutlass::arch::umma_arrive(&s.score_ready);
        }
        for(int page=0;page<16;++page) {
            int slot=page%2;
            cutlass::arch::ClusterBarrier::wait(&s.p_half,page%2);after_sync();
            uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[slot+2])),VLayout{}));
            #pragma unroll
            for(int k=0;k<64;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,k!=0,inst_pv);
            cutlass::arch::ClusterBarrier::wait(&s.p_ready,page%2);after_sync();
            #pragma unroll
            for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,true,inst_pv);
            cutlass::arch::umma_arrive(&s.pv_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
            // own merge (rows 96-127); issue QK(page+1) after the loads gate.
            float minv=s.inv[page%2][row],mlp=s.lp[page%2][row];
            float nm=max_native(m,mlp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=mlp!=-CUDART_INF_F?exp_native(mlp-safe):0.f;
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float values[32],running[32];
                load32(t+maddr+c,values);
                load32(t+128+maddr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                if(c==96&&local%32==0) cutlass::arch::ClusterBarrier::arrive(&s.merge_loads);
                #pragma unroll
                for(int k=0;k<32;k+=2) merge_pair(running[k],running[k+1],values[k],values[k+1],minv,a,b);
                store32(t+128+maddr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();
            m=nm;z=z*a+b;
            if(page+1<16) {
                cutlass::arch::ClusterBarrier::wait(&s.merge_loads,page%2);
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[(page+1)%2],((page+1)/2)%2);after_sync();
                uint64_t dk=descriptor(s.kv[(page+1)%2]);
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready);
            }
        }
        float finv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
        #pragma unroll 1
        for(int c=0;c<128;c+=32) {
            float running[32];load32(t+128+maddr+c,running);cutlass::arch::fence_view_async_tmem_load();
            uint32_t packed[16];
            #pragma unroll
            for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(running[k*2],running[k*2+1],finv);
            #pragma unroll
            for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,c+k*2)))=*reinterpret_cast<uint4*>(packed+k);
        }
        int q=qbase+row/16;
        if(q<stop) p.lse[int64_t(q)*64+head*16+row%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
    }
    before_sync();__syncthreads();
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.kv);
    for(int i=tid;i<2048;i+=256) {
        int row=i/16,d=i%16*8,q=qbase+row/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+row%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,256);
}
} // namespace rtp_llm
