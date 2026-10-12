#pragma once

// Page-split CUTLASS decode partial: one CTA per (token row, kv head, 4-page
// group). Emits the production _q8kv4_combine ABI (BF16 partial, natural-log
// per-page LSE, counts=16) so the merge is the production combine kernel.
namespace rtp_llm {

struct alignas(128) DecodePartialShared {
    alignas(128) uint8_t q[16384],p[16384],kv[4][16384];
    int blocks[1],physical[1];
    alignas(8) uint64_t kv_ready[2],kv_free[2],score_ready,pv_ready,math_free,p_half,p_ready,done;
    uint32_t tmem;
};
static_assert(sizeof(DecodePartialShared)<=232448,"decode partial exceeds SMEM budget");

__global__ __launch_bounds__(256,2) void grouped_cutlass_decode_partial(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<DecodePartialShared*>(smem);
    const int qbase=blockIdx.x,head=blockIdx.y,grp=blockIdx.z;
    const int tid=threadIdx.x,local=tid%128,warp=local/32;
    const int length=p.rowlens[qbase];
    if(tid==0) {
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.pv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.p_half,1);
        cutlass::arch::ClusterBarrier::init(&s.p_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.math_free,1);
        cutlass::arch::ClusterBarrier::init(&s.done,1);
    }
    if(tid==0) {
        const int slot=grp;
        int block=p.topk[head*p.th+int64_t(qbase)*p.tq+slot];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[int64_t(qbase)*p.tb+block]:-1;
        s.blocks[0]=block;
        s.physical[0]=(valid&&physical>=0&&physical<p.pages)?physical:-1;
    }
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
    for(int i=tid;i<1024;i+=256) {
        int row=i/8,d=i%8*16;
        Copy::copy(*reinterpret_cast<const uint4*>(p.q+int64_t(qbase)*p.qn+(head*16+row%16)*p.qh+d),
                   *reinterpret_cast<uint4*>(s.q+Layout{}(cute::make_coord(row,d))),row<16);
    }
    cute::cp_async_fence();cute::cp_async_wait<0>();
    cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    const uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if(tid<128) {
        cutlass::arch::warpgroup_reg_alloc<192>();
        {
            constexpr int page=0;
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,0);after_sync();
            int block=s.blocks[0];
            bool valid=s.physical[0]>=0;
            float inv,lp;
            page_softmax<true>(t,addr,block,length-1,length,valid,p.alpha,inv,lp,&s.p_half,s.p);
            wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
            #pragma unroll 1
            for(int c=0;c<128;c+=32) {
                float values[32];
                load32(t+addr+c,values);
                cutlass::arch::fence_view_async_tmem_load();

                if(local<16) {
                    uint32_t packed[16];
                    #pragma unroll
                    for(int k=0;k<16;++k) packed[k]=normalize_bf16_pair(values[k*2],values[k*2+1],inv);
                    const int64_t base=((int64_t(qbase)*4+head)*16+grp)*16+local;
                    __nv_bfloat16* dst=p.partial+base*128+c;
                    #pragma unroll
                    for(int k=0;k<16;k+=4) *reinterpret_cast<uint4*>(dst+k*2)=*reinterpret_cast<const uint4*>(packed+k);
                }
            }
            if(local<16) {
                const int64_t base=((int64_t(qbase)*4+head)*16+grp)*16+local;
                p.partial_lse[base]=lp==-CUDART_INF_F?-CUDART_INF_F:lp*0.6931471805599453f;
            }
        }
        wg_sync<1>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        if(local<96) {
            {
                constexpr int page=0;
                int slot=0;
                int block=s.blocks[0],physical=s.physical[0];
                bool valid=physical>=0;
                int src=valid?physical:0;
                for(int i=local;i<1024;i+=96) {
                    int token=i/8,g=i%8,d=g*16;
                    int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
                    uint64_t pk=0,pv=0;uint8_t sk=0,sv=0;
                    if(valid&&block*128+token<length) {
                        pk=*reinterpret_cast<const uint64_t*>(p.k+int64_t(src)*p.kp+head*8192+token*64+d/2);
                        pv=*reinterpret_cast<const uint64_t*>(p.v+int64_t(src)*p.vp+head*8192+token*64+d/2);
                        sk=p.ks[int64_t(src)*p.ksp+head*1024+sf];
                        sv=p.vs[int64_t(src)*p.vsp+head*1024+sf];
                    }
                    uint4 key,value;
                    #pragma unroll
                    for(int c=0;c<4;++c) {
                        reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(pk>>(c*16)),sk);
                        reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(pv>>(c*16)),sv);
                    }
                    *reinterpret_cast<uint4*>(s.kv[slot]+Layout{}(cute::make_coord(token,d)))=key;
                    *reinterpret_cast<uint4*>(s.kv[slot+2]+VLayout{}(cute::make_coord(d,token)))=value;
                }
                cutlass::arch::fence_view_async_shared();loader_sync();
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
            }
        } else {
            uint64_t dq=descriptor(s.q),dp=descriptor(s.p);
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            {
                constexpr int page=0;
                int slot=0;
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],0);after_sync();
                uint64_t dk=descriptor(s.kv[0]);
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready);
                cutlass::arch::ClusterBarrier::wait(&s.p_half,page%2);after_sync();
                uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.kv[2])),VLayout{}));
                #pragma unroll
                for(int k=0;k<64;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,k!=0,inst_pv);
                cutlass::arch::ClusterBarrier::wait(&s.p_ready,page%2);after_sync();
                #pragma unroll
                for(int k=64;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t,true,inst_pv);
                cutlass::arch::umma_arrive(&s.pv_ready);
                cutlass::arch::umma_arrive(&s.kv_free[slot]);
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    if(grp==0&&tid==0) p.counts[qbase*4+head]=16;
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(t,256);
}

} // namespace rtp_llm
