#pragma once

namespace rtp_llm {
// M64 TMEM layout probe: QK with known E4M3 inputs, dump dp space from all 8 warps.
__global__ __launch_bounds__(256,1) void m64_layout_probe(const uint8_t* q, const uint8_t* k, float* out) {
    __shared__ alignas(128) uint8_t sq[8192], sk[16384];
    __shared__ alignas(8) uint64_t mma_done;
    __shared__ uint32_t tmem;
    int tid=threadIdx.x;
    if(tid==0) cutlass::arch::ClusterBarrier::init(&mma_done,1);
    for(int i=tid;i<1024;i+=256) {
        int r=i/8,d=i%8*16;
        *reinterpret_cast<uint4*>(sq+Layout64{}(cute::make_coord(r,d)))=*reinterpret_cast<const uint4*>(q+r*128+d);
    }
    for(int i=tid;i<16384;i+=256) sk[i]=0x38;
    if(tid<32) { cute::TMEM::Allocator1Sm().allocate(256,&tmem); cute::TMEM::Allocator1Sm().release_allocation_lock(); }
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if(tid<32) {
        uint64_t dq=descriptor64(sq);
        uint64_t dk=descriptor(sk);
        uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk+kk/16,tmem,kk!=0,inst);
        uint64_t inst64=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,64,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        uint64_t dk_lo=cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(sk)),Layout64{}));
        uint64_t dk_hi=cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(sk+Layout{}(cute::make_coord(64,0)))),Layout64{}));
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk_lo+kk/16,tmem,kk!=0,inst64);
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk_hi+kk/16,tmem+64,kk!=0,inst64);
        cutlass::arch::umma_arrive(&mma_done);
    }
    cutlass::arch::ClusterBarrier::wait(&mma_done,0);
    after_sync();
    int warp=tid/32, lane=tid%32;
    {
        float v[32];
        uint32_t addr=tmem+(uint32_t(warp*32)<<16);
        #pragma unroll
        for(int c=0;c<128;c+=32) load32(addr+c,v+c);
        cutlass::arch::fence_view_async_tmem_load();
        for(int c=0;c<128;++c) out[(warp*32+lane)*128+c]=v[c];
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(tmem,256);
}


// Decode the (row,col) mapping of 16dp32b64x loads for M64 D layout.
__global__ __launch_bounds__(256,1) void m64_loadmap_probe(const uint8_t* q, const uint8_t* k, float* out) {
    __shared__ alignas(128) uint8_t sq[8192], sk[16384];
    __shared__ alignas(8) uint64_t mma_done;
    __shared__ uint32_t tmem;
    int tid=threadIdx.x;
    if(tid==0) cutlass::arch::ClusterBarrier::init(&mma_done,1);
    for(int i=tid;i<1024;i+=256) {
        int r=i/8,d=i%8*16;
        *reinterpret_cast<uint4*>(sq+Layout64{}(cute::make_coord(r,d)))=*reinterpret_cast<const uint4*>(q+r*128+d);
    }
    for(int i=tid;i<1024;i+=256) {
        int t=i/128,d=i%128;
        sk[Layout{}(cute::make_coord(t,(d/16)*16))+ (d%16)]=k[t*128+d];
    }
    if(tid<32) { cute::TMEM::Allocator1Sm().allocate(256,&tmem); cute::TMEM::Allocator1Sm().release_allocation_lock(); }
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if(tid<32) {
        uint64_t dq=descriptor64(sq);
        uint64_t dk=descriptor(sk);
        uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk+kk/16,tmem,kk!=0,inst);
        uint64_t inst64=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,64,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        uint64_t dk_lo=cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(sk)),Layout64{}));
        uint64_t dk_hi=cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(sk+Layout{}(cute::make_coord(64,0)))),Layout64{}));
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk_lo+kk/16,tmem,kk!=0,inst64);
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk_hi+kk/16,tmem+64,kk!=0,inst64);
        cutlass::arch::umma_arrive(&mma_done);
    }
    cutlass::arch::ClusterBarrier::wait(&mma_done,0);
    after_sync();
    int warp=tid/32, lane=tid%32;
    if(warp<2) {
        // load32 (32dp, resident-proven column addressing): warp reads its own dp window
        float v[128];
        #pragma unroll
        for(int c=0;c<128;c+=32) load32(tmem+(uint32_t(warp*32)<<16)+c,v+c);
        cutlass::arch::fence_view_async_tmem_load();
        for(int i=0;i<128;++i) out[(warp*32+lane)*128+i]=v[i];
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(tmem,256);
}

// Probe: tcgen05.st from warps 4-7 — do stores land in the warp's own dp window?
__global__ __launch_bounds__(256,1) void warp_store_probe(float* out) {
    __shared__ uint32_t tmem;
    __shared__ alignas(8) uint64_t mma_done;
    int tid=threadIdx.x;
    if(tid==0) cutlass::arch::ClusterBarrier::init(&mma_done,1);
    if(tid<32) { cute::TMEM::Allocator1Sm().allocate(256,&tmem); cute::TMEM::Allocator1Sm().release_allocation_lock(); }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    int warp=tid/32, lane=tid%32;
    // every warp stores value (warp*1000+lane) at its own dp window, cols 0..127 (4 chunks)
    if(warp<8) {
        uint32_t vals[32];
        #pragma unroll
        for(int i=0;i<32;++i) vals[i]=__float_as_uint(float(warp*1000+lane)+float(i)*0.001f);
        #pragma unroll
        for(int c=0;c<128;c+=32) store32(tmem+(uint32_t(warp*32)<<16)+c,vals);
        cutlass::arch::fence_view_async_tmem_store();
    }
    before_sync();__syncthreads();
    // read back with load32 from warps 0-3 (all 128 dp)
    if(warp<4) {
        float v[128];
        #pragma unroll
        for(int c=0;c<128;c+=32) load32(tmem+(uint32_t(warp*32)<<16)+c,v+c);
        cutlass::arch::fence_view_async_tmem_load();
        for(int i=0;i<128;++i) out[(warp*32+lane)*128+i]=v[i];
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(tmem,256);
}

// 2-CTA TMEM isolation: both CTAs store+read their own TMEM concurrently.
__global__ __launch_bounds__(256,2) void warp_store_probe2(float* out) {
    __shared__ uint32_t tmem;
    __shared__ alignas(8) uint64_t mma_done;
    int tid=threadIdx.x;
    int cta=blockIdx.x;
    if(tid==0) cutlass::arch::ClusterBarrier::init(&mma_done,1);
    if(tid<32) { cute::TMEM::Allocator1Sm().allocate(256,&tmem); cute::TMEM::Allocator1Sm().release_allocation_lock(); }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    int warp=tid/32, lane=tid%32;
    // each warp of each CTA stores cta*100000+warp*1000+lane into its dp window
    if(warp<8) {
        uint32_t vals[32];
        #pragma unroll
        for(int i=0;i<32;++i) vals[i]=__float_as_uint(float(cta*100000+warp*1000+lane)+float(i)*0.001f);
        #pragma unroll
        for(int c=0;c<128;c+=32) store32(tmem+(uint32_t(warp*32)<<16)+c,vals);
        cutlass::arch::fence_view_async_tmem_store();
    }
    before_sync();__syncthreads();
    // read back with load32 from warps 0-3 AND 4-7 (both warpgroups)
    {
        float v[128];
        #pragma unroll
        for(int c=0;c<128;c+=32) load32(tmem+(uint32_t(warp*32)<<16)+c,v+c);
        cutlass::arch::fence_view_async_tmem_load();
        for(int i=0;i<128;++i) out[(cta*256+tid)*128+i]=v[i];
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(tmem,256);
}

// Split-QK probe: N=128 full QK vs two N=64 halves (B rows 0-63 / 64-127).
__global__ __launch_bounds__(128,1) void n64_split_qk_probe(const uint8_t* q, const uint8_t* k, float* out) {
    __shared__ alignas(128) uint8_t sq[16384], sk[16384];
    __shared__ alignas(8) uint64_t mma_done;
    __shared__ uint32_t tmem;
    int tid=threadIdx.x, lane=tid%32;
    if(tid==0) cutlass::arch::ClusterBarrier::init(&mma_done,1);
    for(int i=tid;i<1024;i+=128) {
        int r=i/8,d=i%8*16;
        *reinterpret_cast<uint4*>(sq+Layout{}(cute::make_coord(r,d)))=*reinterpret_cast<const uint4*>(q+r*128+d);
    }
    for(int i=tid;i<1024;i+=128) {
        int t=i/8,d=i%8*16;
        *reinterpret_cast<uint4*>(sk+Layout{}(cute::make_coord(t,d)))=*reinterpret_cast<const uint4*>(k+t*128+d);
    }
    if(tid<32) { cute::TMEM::Allocator1Sm().allocate(256,&tmem); cute::TMEM::Allocator1Sm().release_allocation_lock(); }
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if(tid<32) {
        if(tid==0) {
            printf("off(64,0)=%d off(96,0)=%d off(0,64)=%d\n",
                (int)Layout{}(cute::make_coord(64,0)),(int)Layout{}(cute::make_coord(96,0)),(int)Layout{}(cute::make_coord(0,64)));
        }
        uint64_t dq=descriptor(sq);
        uint64_t inst128=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,descriptor(sk)+kk/16,tmem+128,kk!=0,inst128);
        uint64_t inst64=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,64,cute::UMMA::Major::K,cute::UMMA::Major::K>();
        uint64_t dk_lo=cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(sk)),Layout64{}));
        uint64_t dk_hi=cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(sk+Layout{}(cute::make_coord(64,0)))),Layout64{}));
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk_lo+kk/16,tmem,kk!=0,inst64);
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dk_hi+kk/16,tmem+64,kk!=0,inst64);
        cutlass::arch::umma_arrive(&mma_done);
    }
    cutlass::arch::ClusterBarrier::wait(&mma_done,0);
    after_sync();
    // dump both halves and the reference
    if(tid<64) {
        float v[128];
        if(tid<32) {
            #pragma unroll
            for(int c=0;c<128;c+=32) load32(tmem+c,v+c);
            cutlass::arch::fence_view_async_tmem_load();
            for(int i=0;i<128;++i) out[lane*128+i]=v[i];
        } else {
            #pragma unroll
            for(int c=0;c<128;c+=32) load32(tmem+128+c,v+c);
            cutlass::arch::fence_view_async_tmem_load();
            for(int i=0;i<128;++i) out[4096+lane*128+i]=v[i];
        }
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(tmem,256);
}

// PV N=64 D-split probe: full N=128 PV vs two N=64 PVs (V in two 64-wide
// MN-major tiles), all vs CPU.
__global__ __launch_bounds__(128,1) void pv_split_probe(const uint8_t* q, const uint8_t* v, float* out) {
    __shared__ alignas(128) uint8_t sq[16384];
    __shared__ alignas(128) uint8_t vlo[8192], vhi[8192];
    __shared__ alignas(128) uint8_t vfull[16384];
    __shared__ alignas(8) uint64_t mma_done;
    __shared__ uint32_t tmem;
    int tid=threadIdx.x, lane=tid%32;
    if(tid==0) cutlass::arch::ClusterBarrier::init(&mma_done,1);
    for(int i=tid;i<1024;i+=128) {
        int r=i/8,d=i%8*16;
        *reinterpret_cast<uint4*>(sq+Layout{}(cute::make_coord(r,d)))=*reinterpret_cast<const uint4*>(q+r*128+d);
    }
    // full V as 128x128 MN-major
    for(int i=tid;i<1024;i+=128) {
        int t=i/8,d=i%8*16;
        *reinterpret_cast<uint4*>(vfull+VLayout{}(cute::make_coord(d,t)))=*reinterpret_cast<const uint4*>(v+t*128+d);
        // halves as 64x128 MN-major tiles (d 0-63 -> vlo, 64-127 -> vhi)
        if(d<64) *reinterpret_cast<uint4*>(vlo+VLayout64MN{}(cute::make_coord(d,t)))=*reinterpret_cast<const uint4*>(v+t*128+d);
        else *reinterpret_cast<uint4*>(vhi+VLayout64MN{}(cute::make_coord(d-64,t)))=*reinterpret_cast<const uint4*>(v+t*128+d);
    }
    if(tid<32) { cute::TMEM::Allocator1Sm().allocate(256,&tmem); cute::TMEM::Allocator1Sm().release_allocation_lock(); }
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if(tid<32) {
        uint64_t dq=descriptor(sq);
        uint64_t dvfull=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(vfull)),VLayout{}));
        uint64_t dvlo=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(vlo)),VLayout64MN{}));
        uint64_t dvhi=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(vhi)),VLayout64MN{}));
        uint64_t inst128=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
        uint64_t inst64=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,64,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
        // reference: full N=128 into cols 128..255
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dvfull+VLayout{}(cute::make_coord(0,kk))/16,tmem+128,kk!=0,inst128);
        // split: lo into cols 0..63, hi into cols 64..127
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dvlo+VLayout64MN{}(cute::make_coord(0,kk))/16,tmem,kk!=0,inst64);
        for(int kk=0;kk<128;kk+=32) QKMma::fma(dq+kk/16,dvhi+VLayout64MN{}(cute::make_coord(0,kk))/16,tmem+64,kk!=0,inst64);
        cutlass::arch::umma_arrive(&mma_done);
    }
    cutlass::arch::ClusterBarrier::wait(&mma_done,0);
    after_sync();
    if(tid<64) {
        float x[128];
        if(tid<32) {
            #pragma unroll
            for(int c=0;c<128;c+=32) load32(tmem+c,x+c);
            cutlass::arch::fence_view_async_tmem_load();
            for(int i=0;i<128;++i) out[lane*128+i]=x[i];
        } else {
            #pragma unroll
            for(int c=0;c<128;c+=32) load32(tmem+128+c,x+c);
            cutlass::arch::fence_view_async_tmem_load();
            for(int i=0;i<128;++i) out[4096+lane*128+i]=x[i];
        }
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(tmem,256);
}
} // namespace rtp_llm
