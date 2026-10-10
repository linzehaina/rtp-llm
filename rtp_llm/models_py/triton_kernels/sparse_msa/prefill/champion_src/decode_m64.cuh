#pragma once

// decode_m64: M64 decode attention with 4-page groups and register
// accumulators, mirroring the production four_page_partial structure.
// Grid: (rows, 4 kv heads, 4 page-groups). One CTA processes 4 pages for
// one (token row, head) with M64 UMMA: 16 live q-head rows at dp 0-15
// (m64_probe layout), all 128 KV tokens as N, 128 dims as PV N.
// Running O in registers on the 16 live lanes of warp 0.

namespace rtp_llm {

struct alignas(128) DecodeM64Shared {
    alignas(128) uint8_t q[8192],p[8192],k[16384],v[16384];
    int blocks[4],physical[4];
    float inv[4][16],lp[4][16];
    alignas(8) uint64_t kv_ready,kv_free,score_ready,pv_ready,p_ready;
    uint32_t tmem;
};

__global__ __launch_bounds__(256,1) void grouped_cutlass_decode_m64(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<DecodeM64Shared*>(smem);
    const int qrow=blockIdx.x, head=blockIdx.y, grp=blockIdx.z;
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int length=p.rowlens[qrow];
    if(tid==0) {
        cutlass::arch::ClusterBarrier::init(&s.kv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.kv_free,1);
        cutlass::arch::ClusterBarrier::init(&s.score_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.pv_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.p_ready,1);
    }
    if(tid<4) {
        const int slot=grp*4+tid;
        int block=p.topk[head*p.th+int64_t(qrow)*p.tq+slot];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int physical=valid?p.table[int64_t(qrow)*p.tb+block]:-1;
        s.blocks[tid]=block;
        s.physical[tid]=(valid&&physical>=0&&physical<p.pages)?physical:-1;
    }
    if(tid<32) {cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
    for(int i=tid;i<2048;i+=256) {
        int r=i/32,d=i%32*4;
        uint32_t x=0;
        if(r<16) x=*reinterpret_cast<const uint32_t*>(p.q+int64_t(qrow)*p.qn+(head*16+r)*p.qh+d);
        *reinterpret_cast<uint32_t*>(s.q+Layout64{}(cute::make_coord(r,d)))=x;
    }
    for(int i=tid;i<2048;i+=256) *reinterpret_cast<uint32_t*>(s.p+i*4)=0;
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    const uint32_t t=s.tmem;
    if(warp==0) {
        // warp 0 only: lanes 0-15 are the live M64 rows (dp 0-15).
        const bool live=lane<16;
        float m=-CUDART_INF_F,z=0.f;
        float run[128];
        #pragma unroll
        for(int i=0;i<128;++i) run[i]=0.f;
        const int last0=length-1;
        for(int page=0;page<4;++page) {
            cutlass::arch::ClusterBarrier::wait(&s.score_ready,page%2);after_sync();
            int block=s.blocks[page];
            bool valid=s.physical[page]>=0;
            int last=valid?last0-block*128:-1;
            float maximum=-CUDART_INF_F;
            if(live) {
                #pragma unroll
                for(int c=0;c<128;c+=32) {
                    float v[32];
                    load32(t+c,v);
                    cutlass::arch::fence_view_async_tmem_load();
                    #pragma unroll
                    for(int i=0;i<32;++i) {
                        int col=c+i;
                        if(col<=last) maximum=max_native(maximum,v[i]);
                    }
                }
            }
            float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
            float den=0.f;
            if(live) {
                #pragma unroll
                for(int c=0;c<128;c+=32) {
                    float v[32];
                    load32(t+c,v);
                    cutlass::arch::fence_view_async_tmem_load();
                    #pragma unroll
                    for(int g=0;g<8;++g) {
                        float e[4];
                        #pragma unroll
                        for(int i=0;i<4;++i) {
                            int col=c+g*4+i;
                            float x=(col<=last)?v[g*4+i]:0.f;
                            x=fmaf(x,p.alpha,shift);
                            x=(col>=32&&col<96&&col%16>=12)?exp_emulated(x):exp_native(x);
                            e[i]=x;
                            den+=x;
                        }
                        *reinterpret_cast<uint32_t*>(s.p+Layout64{}(cute::make_coord(lane,c+g*4)))=pack_probability(e[0],e[1],e[2],e[3]);
                    }
                }
            }
            float inv=rcp_native(den>0.f?den:1.f);
            float lp=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
            if(live) { s.inv[page][lane]=inv; s.lp[page][lane]=lp; }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if(lane==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            cutlass::arch::ClusterBarrier::wait(&s.pv_ready,page%2);after_sync();
            float nm=max_native(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            if(live) {
                #pragma unroll
                for(int c=0;c<128;c+=32) {
                    float values[32];
                    load32(t+128+c,values);
                    cutlass::arch::fence_view_async_tmem_load();
                    #pragma unroll
                    for(int k=0;k<32;k+=2) merge_pair(run[c+k],run[c+k+1],values[k],values[k+1],inv,a,b);
                }
            }
            m=nm;z=z*a+b;
        }
        if(live) {
            float inv=rcp_native(z>0.f?z:1.f);
            #pragma unroll
            for(int d=0;d<128;d+=8) {
                uint32_t pk[4];
                #pragma unroll
                for(int k=0;k<4;++k) pk[k]=normalize_bf16_pair(run[d+k*2],run[d+k*2+1],inv);
                __nv_bfloat16* outrow=p.out+(int64_t(qrow)*64+head*16+lane)*128+d;
                *reinterpret_cast<uint4*>(outrow)=*reinterpret_cast<const uint4*>(pk);
            }
            p.lse[int64_t(qrow)*64+head*16+lane]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
        }
        __syncwarp();
    } else if(tid>=128 && tid<224) {
        // loader warps 4-6 (96 threads).
        const int llocal=tid-128;
        for(int page=0;page<4;++page) {
            if(page>0) cutlass::arch::ClusterBarrier::wait(&s.kv_free,(page-1)%2);
            int block=s.blocks[page],physical=s.physical[page];
            bool valid=physical>=0;
            int src=valid?physical:0;
            for(int i=llocal;i<1024;i+=96) {
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
                *reinterpret_cast<uint4*>(s.k+Layout{}(cute::make_coord(token,d)))=key;
                *reinterpret_cast<uint4*>(s.v+VLayout{}(cute::make_coord(d,token)))=value;
            }
            cutlass::arch::fence_view_async_shared();
            asm volatile("bar.sync 4, 96;" ::: "memory");
            if(llocal==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready);
        }
    } else {
        // MMA warp 7.
        uint64_t dq=descriptor64(s.q),dp=descriptor64(s.p);
        uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
        for(int page=0;page<4;++page) {
            cutlass::arch::ClusterBarrier::wait(&s.kv_ready,page%2);after_sync();
            uint64_t dk=descriptor(s.k);
            #pragma unroll
            for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
            cutlass::arch::umma_arrive(&s.score_ready);
            cutlass::arch::ClusterBarrier::wait(&s.p_ready,page%2);after_sync();
            uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.v)),VLayout{}));
            #pragma unroll
            for(int k=0;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+128,k!=0,inst_pv);
            cutlass::arch::umma_arrive(&s.pv_ready);
            if(page<3) cutlass::arch::umma_arrive(&s.kv_free);
        }
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(t,256);
}

} // namespace rtp_llm
