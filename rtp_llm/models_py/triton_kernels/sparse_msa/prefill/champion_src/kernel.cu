#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <utility>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/float8.h>
#include <cutlass/arch/reg_reconfig.h>

#ifndef MSA_TRACE
#define MSA_TRACE 0
#endif
#ifndef MSA_DUAL
#define MSA_DUAL 0
#endif
#ifndef MSA_DECODED
#define MSA_DECODED 0
#endif
#ifndef MSA_LATE_MERGE
#define MSA_LATE_MERGE 0
#endif
#ifndef MSA_COMPACT
#define MSA_COMPACT 0
#endif
#ifndef MSA_DUAL_QUERY
#define MSA_DUAL_QUERY 0
#endif
#ifndef MSA_SPLIT_MERGE
#define MSA_SPLIT_MERGE 0
#endif
#ifndef MSA_RESIDENT
#define MSA_RESIDENT 0
#endif
#ifndef MSA_RESIDENT2
#define MSA_RESIDENT2 0
#endif
#ifndef MSA_RESIDENT3
#define MSA_RESIDENT3 0
#endif
#ifndef MSA_RESIDENT4
#define MSA_RESIDENT4 0
#endif
#ifndef MSA_RESIDENT5
#define MSA_RESIDENT5 0
#endif
#ifndef MSA_RESIDENT6
#define MSA_RESIDENT6 0
#endif
#ifndef MSA_RESIDENT7
#define MSA_RESIDENT7 0
#endif
#ifndef MSA_RESIDENT8
#define MSA_RESIDENT8 0
#endif
#ifndef MSA_RESIDENT9
#define MSA_RESIDENT9 0
#endif
#ifndef MSA_RESIDENT11
#define MSA_RESIDENT11 0
#endif
#ifndef MSA_RESIDENT12
#define MSA_RESIDENT12 0
#endif
#ifndef MSA_RESIDENT13
#define MSA_RESIDENT13 0
#endif
#ifndef MSA_RESIDENT14
#define MSA_RESIDENT14 0
#endif

namespace rtp_llm {
__device__ int g_wide_thresh=140;
using F8 = cutlass::float_e4m3_t;
using Layout = decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<F8>{},
                                           cute::Shape<cute::_128, cute::_128>{}));
using QKMma = cute::SM100_MMA_F8F6F4_SS;
using PVMma = cute::SM100_MMA_F8F6F4_TS<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>;
using PVMmaMN = cute::SM100_MMA_F8F6F4_TS<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>;
using VLayout = decltype(cute::tile_to_shape(cute::UMMA::Layout_MN_SW128_Atom<F8>{},
                                            cute::Shape<cute::_128,cute::_128>{}));
using VLayout64MN = decltype(cute::tile_to_shape(cute::UMMA::Layout_MN_SW64_Atom<F8>{},
                                                 cute::Shape<cute::_64,cute::_128>{}));

struct Params {
    const uint8_t *q, *k, *v, *ks, *vs;
    const int32_t *topk, *table, *cuq, *cuk, *used;
    __nv_bfloat16* out;
    float* lse;
    int64_t qn, qh, kp, vp, th, tq, tb, ksp, vsp;
    int pages, max_pages, max_q, batches;
    float alpha;
    uint64_t* debug;
    uint8_t *k8,*v8;
    const int32_t* rowlens;
    __nv_bfloat16* partial;
    float* partial_lse;
    int32_t* counts;
};

struct alignas(128) Shared {
    alignas(128) uint8_t q[16384], k[16384], v[16384];
    alignas(8) uint64_t done;
    uint32_t tmem;
};

__device__ __forceinline__ float exp_native(float x) {
    float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float log_native(float x) {
    float r; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp_native(float x) {
    float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float max_native(float a,float b) {
    float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a),"f"(b)); return r;
}
// Programmatic dependent launch (PDL): the dependent kernel is launched with
// cudaLaunchAttributeProgrammaticStreamSerialization, so its grid may launch
// while the primary still runs. grid_dep_wait blocks until the primary's
// memory is complete/visible (so it must precede any read of primary-written
// data); grid_dep_launch only gates the dependent's SCHEDULING, so its
// placement is a performance choice and never a correctness one.
__device__ __forceinline__ void grid_dep_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void grid_dep_launch() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ uint32_t pack_probability(float a,float b,float c,float d) {
    uint32_t r;
    asm("{ .reg .b16 x,y; cvt.rn.satfinite.e4m3x2.f32 x,%2,%1; "
        "cvt.rn.satfinite.e4m3x2.f32 y,%4,%3; mov.b32 %0,{x,y}; }"
        : "=r"(r) : "f"(a),"f"(b),"f"(c),"f"(d));
    return r;
}
__device__ __forceinline__ void add_pair(float& a,float& b,float x,float y) {
    asm("{ .reg .b64 ab,xy,r; mov.b64 ab,{%0,%1}; mov.b64 xy,{%2,%3}; "
        "add.rn.f32x2 r,ab,xy; mov.b64 {%0,%1},r; }"
        : "+f"(a),"+f"(b) : "f"(x),"f"(y));
}
__device__ __forceinline__ void affine_pair(float& x,float& y,float scale,float shift) {
    asm("{ .reg .b64 xy,s,t,r; mov.b64 xy,{%0,%1}; mov.b64 s,{%2,%2}; mov.b64 t,{%3,%3}; "
        "fma.rn.f32x2 r,xy,s,t; mov.b64 {%0,%1},r; }"
        : "+f"(x),"+f"(y) : "f"(scale),"f"(shift));
}
__device__ __forceinline__ void merge_pair(float& old0,float& old1,float p0,float p1,float inv,float a,float b) {
    asm("{ .reg .b64 x,y,s,t,r; .reg .b32 lo,hi,bf; "
        "mov.b64 x,{%2,%3}; mov.b64 s,{%4,%4}; mul.rn.f32x2 x,x,s; "
        "mov.b64 {lo,hi},x; cvt.rn.bf16x2.f32 bf,hi,lo; "
        "shl.b32 lo,bf,16; and.b32 hi,bf,0xffff0000; mov.b64 x,{lo,hi}; "
        "mov.b64 y,{%0,%1}; mov.b64 s,{%5,%5}; mov.b64 t,{%6,%6}; "
        "mul.rn.f32x2 y,y,s; mul.rn.f32x2 x,x,t; add.rn.f32x2 r,y,x; "
        "mov.b64 {%0,%1},r; }"
        : "+f"(old0),"+f"(old1) : "f"(p0),"f"(p1),"f"(inv),"f"(a),"f"(b));
}
__device__ __forceinline__ float exp_emulated(float x) {
    float rounded;
    x=fmaxf(x,-127.f);
    asm("add.rm.ftz.f32 %0, %1, %2;" : "=f"(rounded) : "f"(x), "f"(12582912.f));
    float f=fmaf(-(rounded-12582912.f),1.f,x);
    float y=fmaf(0.07711908966302872f,f,0.22756439447402954f);
    y=fmaf(y,f,0.6951461434364319f);
    y=fmaf(y,f,1.f);
    return __uint_as_float((__float_as_uint(rounded)<<23)+__float_as_uint(y));
}

__device__ __forceinline__ uint32_t decode4(uint16_t p, uint8_t s) {
    uint32_t r;
    asm("{ .reg .b8 b0,b1,s0,s1; .reg .b16 sf,e0,e1; .reg .b32 sf2,h0,h1;\n"
        "mov.b16 {b0,b1}, %1; mov.b16 {s0,s1}, %2; mov.b16 sf,{s0,s0};\n"
        "cvt.rn.f16x2.e4m3x2 sf2,sf;\n"
        "cvt.rn.f16x2.e2m1x2 h0,b0; cvt.rn.f16x2.e2m1x2 h1,b1;\n"
        "mul.rn.f16x2 h0,h0,sf2; mul.rn.f16x2 h1,h1,sf2;\n"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0,h0; cvt.rn.satfinite.e4m3x2.f16x2 e1,h1;\n"
        "mov.b32 %0,{e0,e1}; }" : "=r"(r) : "h"(p),"h"(uint16_t(s)));
    return r;
}

__global__ void decode_cache_coop(Params p, uint8_t* mark) {
    // Phase 1: coalesced flat scan of the topk array -> mark bitmap.
    // Phase 2 (after grid sync): the decode_working_cache body.
    const int tiles=(p.max_q+7)/8;
    const int64_t total_entries=int64_t(p.batches)*4*tiles*16;
    for(int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; idx<total_entries; idx+=int64_t(gridDim.x)*blockDim.x) {
        int slot=idx%16, tile=(idx/16)%tiles, h=(idx/16/tiles)%4, batch=idx/16/tiles/4;
        int64_t qbase=p.cuq[batch]+int64_t(tile)*8;
        if(qbase>=p.cuq[batch+1]) continue;
        int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
        int block=p.topk[h*p.th+qbase*p.tq+slot];
        if(block<0||block>=p.max_pages||block*128>=length) continue;
        int phys=p.table[int64_t(batch)*p.tb+block];
        if(phys>=0&&phys<p.pages) reinterpret_cast<uint32_t*>(mark)[int64_t(phys)*4+h]=1;
    }
    cooperative_groups::this_grid().sync();
    const int page=blockIdx.x,head=blockIdx.y;
    if(!__ldcg(reinterpret_cast<const uint32_t*>(mark)+int64_t(page)*4+head)) return;
    for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
        int token=i/8,g=i%8,d=g*16;
        int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
        int64_t so=int64_t(page)*4096+head*1024+sf;
        uint64_t pk=*reinterpret_cast<const uint64_t*>(p.k+int64_t(page)*p.kp+head*8192+token*64+d/2);
        uint64_t pv=*reinterpret_cast<const uint64_t*>(p.v+int64_t(page)*p.vp+head*8192+token*64+d/2);
        uint8_t sk=p.ks[so],sv=p.vs[so];
        uint4 key,value;
        #pragma unroll
        for(int c=0;c<4;++c) {
            reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(pk>>(c*16)),sk);
            reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(pv>>(c*16)),sv);
        }
        int64_t offset=(int64_t(page)*4+head)*16384+token*128+d;
        *reinterpret_cast<uint4*>(p.k8+offset)=key;
        *reinterpret_cast<uint4*>(p.v8+offset)=value;
    }
    if(threadIdx.x==0) reinterpret_cast<uint32_t*>(mark)[int64_t(page)*4+head]=0;
}

__global__ void decode_cache_selfmark(Params p) {
    // Fused mark+decode: each (page, head) CTA scans the (L2-resident) topk
    // lists to decide whether its page was selected by any tile, then
    // decodes if so. Removes the memset + mark graph nodes.
    const int page=blockIdx.x,head=blockIdx.y;
    // scan: any (batch, head', tile, slot) whose resolved physical page == page
    bool selected=false;
    const int tiles=(p.max_q+7)/8;
    for(int idx=threadIdx.x; idx<p.batches*4*tiles*16 && !selected; idx+=blockDim.x) {
        int slot=idx%16, tile=(idx/16)%tiles, h=(idx/16/tiles)%4, batch=idx/16/tiles/4;
        int64_t qbase=p.cuq[batch]+int64_t(tile)*8;
        if(qbase>=p.cuq[batch+1]) continue;
        int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
        int block=p.topk[h*p.th+qbase*p.tq+slot];
        if(block<0||block>=p.max_pages||block*128>=length) continue;
        int phys=p.table[int64_t(batch)*p.tb+block];
        if(phys==page) selected=true;
    }
    // block-wide OR
    __shared__ int sel_sh;
    if(threadIdx.x==0) sel_sh=0;
    __syncthreads();
    if(selected) atomicOr(&sel_sh,1);
    __syncthreads();
    if(!sel_sh) return;
    for(int i=threadIdx.x;i<1024;i+=blockDim.x) {
        int token=i/8,g=i%8,d=g*16;
        int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
        int64_t so=int64_t(page)*4096+head*1024+sf;
        uint64_t pk=*reinterpret_cast<const uint64_t*>(p.k+int64_t(page)*p.kp+head*8192+token*64+d/2);
        uint64_t pv=*reinterpret_cast<const uint64_t*>(p.v+int64_t(page)*p.vp+head*8192+token*64+d/2);
        uint8_t sk=p.ks[so],sv=p.vs[so];
        uint4 key,value;
        #pragma unroll
        for(int c=0;c<4;++c) {
            reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(pk>>(c*16)),sk);
            reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(pv>>(c*16)),sv);
        }
        int64_t offset=(int64_t(page)*4+head)*16384+token*128+d;
        *reinterpret_cast<uint4*>(p.k8+offset)=key;
        *reinterpret_cast<uint4*>(p.v8+offset)=value;
    }
}

__global__ void mark_selected_pages(Params p, uint8_t* mark) {
    // one thread per (head, tile, slot): sets mark[page] for every page any
    // tile selected. Duplicate marks are idempotent byte writes.
    // Buffer layout after the marks: [epoch u32][stamps pages*4 u32].
    // The epoch identifies this run; bumping it invalidates every stale
    // stamp from previous runs (loader spins on stamp == current epoch).
    if(blockIdx.x==0&&threadIdx.x==0)
        ++*reinterpret_cast<volatile uint32_t*>(mark+size_t(p.pages)*16);
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int64_t total=int64_t(p.batches)*4*((p.max_q+7)/8)*16;
    if(idx>=total) { grid_dep_launch(); return; }
    int slot=idx%16, tile=(idx/16)%((p.max_q+7)/8), head=(idx/16/((p.max_q+7)/8))%4;
    int64_t batch=idx/16/((p.max_q+7)/8)/4;
    int64_t qbase=p.cuq[batch]+int64_t(tile)*8;
    if(qbase>=p.cuq[batch+1]) { grid_dep_launch(); return; }
    int block=p.topk[head*p.th+qbase*p.tq+slot];
    if(block<0||block>=p.max_pages) { grid_dep_launch(); return; }
    int page=p.table[batch*p.tb+block];
    if(page>=0&&page<p.pages) {
        int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
        int64_t qlast=min(qbase+7,int64_t(p.cuq[batch+1])-1);
        int64_t causal_last=length-(p.cuq[batch+1]-1-qlast);
        if(block*128<length&&block*128<causal_last)
            reinterpret_cast<uint32_t*>(mark)[int64_t(page)*4+head]=1;
    }
    // Trigger AFTER the mark stores: griddepcontrol.wait in the dependent
    // kernel only orders against memory operations issued BEFORE the
    // trigger, so an entry-position trigger would race every mark write.
    grid_dep_launch();
}

__global__ void mark_selected_pages4(Params p, uint8_t* mark) {
    // 4 entries per thread: same semantics, ~4x fewer CTAs so the tail
    // trigger (and the PDL launch of decode_working_cache) happens sooner.
    if(blockIdx.x==0&&threadIdx.x==0)
        ++*reinterpret_cast<volatile uint32_t*>(mark+size_t(p.pages)*16);
    const int tiles=(p.max_q+7)/8;
    const int64_t total=int64_t(p.batches)*4*tiles*16;
    int64_t idx0=int64_t(blockIdx.x)*blockDim.x*4+threadIdx.x;
    #pragma unroll
    for(int e=0;e<4;++e) {
        int64_t idx=idx0+int64_t(e)*blockDim.x;
        if(idx>=total) { if(e==0) grid_dep_launch(); continue; }
        int slot=idx%16, tile=(idx/16)%tiles, head=(idx/16/tiles)%4;
        int64_t batch=idx/16/tiles/4;
        int64_t qbase=p.cuq[batch]+int64_t(tile)*8;
        if(qbase>=p.cuq[batch+1]) continue;
        int block=p.topk[head*p.th+qbase*p.tq+slot];
        if(block<0||block>=p.max_pages) continue;
        int page=p.table[batch*p.tb+block];
        if(page>=0&&page<p.pages) {
            int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
            int64_t qlast=min(qbase+7,int64_t(p.cuq[batch+1])-1);
            int64_t causal_last=length-(p.cuq[batch+1]-1-qlast);
            if(block*128<length&&block*128<causal_last)
                reinterpret_cast<uint32_t*>(mark)[int64_t(page)*4+head]=1;
        }
    }
    grid_dep_launch();
}

__global__ void mark_selected_pages_list(Params p, uint8_t* mark, int cap) {
    // Compact-list variant of mark1: identical (head,tile,slot) entry walk,
    // but the first setter of each (page,head) also appends it to a deduped
    // list so the cache grid covers only selected pairs instead of every
    // page in the pool. Ownership is won with atomicExch on the mark byte
    // (old==0 => this thread is the first setter); the append position comes
    // from a CTA-aggregated counter (one global atomic per CTA) so the
    // counter never serializes thousands of appends.
    if(blockIdx.x==0&&threadIdx.x==0)
        ++*reinterpret_cast<volatile uint32_t*>(mark+size_t(p.pages)*16);
    int32_t* list=reinterpret_cast<int32_t*>(mark+size_t(p.pages)*32+4);
    uint32_t* counter=reinterpret_cast<uint32_t*>(mark+size_t(p.pages)*48+4);
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int64_t total=int64_t(p.batches)*4*((p.max_q+7)/8)*16;
    int entry=-1;
    if(idx<total) {
        int slot=idx%16, tile=(idx/16)%((p.max_q+7)/8), head=(idx/16/((p.max_q+7)/8))%4;
        int64_t batch=idx/16/((p.max_q+7)/8)/4;
        int64_t qbase=p.cuq[batch]+int64_t(tile)*8;
        if(qbase<p.cuq[batch+1]) {
            int block=p.topk[head*p.th+qbase*p.tq+slot];
            if(block>=0&&block<p.max_pages) {
                int page=p.table[batch*p.tb+block];
                if(page>=0&&page<p.pages) {
                    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
                    int64_t qlast=min(qbase+7,int64_t(p.cuq[batch+1])-1);
                    int64_t causal_last=length-(p.cuq[batch+1]-1-qlast);
                    if(block*128<length&&block*128<causal_last) {
                        // one dedicated u32 word per (page,head): first-setter
                        // ownership is a single atomicExch, no CAS retry loop
                        // (the packed-byte layout serialized 4 heads per word
                        // and retried, costing ~2.9us at b4).
                        unsigned int* w=reinterpret_cast<unsigned int*>(mark)+int64_t(page)*4+head;
                        if(atomicExch(w,1u)==0) entry=page*4+head;
                    }
                }
            }
        }
    }
    __shared__ int s_cnt,s_base;
    if(threadIdx.x==0) s_cnt=0;
    __syncthreads();
    int rank=-1;
    if(entry>=0) rank=atomicAdd(&s_cnt,1);
    __syncthreads();
    if(threadIdx.x==0) s_base=atomicAdd(counter,(unsigned)s_cnt);
    __syncthreads();
    if(entry>=0) {
        unsigned long long pos=((unsigned long long)s_base+(unsigned long long)rank)%(unsigned long long)cap;
        list[(int)pos]=entry;
    }
    grid_dep_launch();
}

__global__ void decode_working_cache(Params p, uint8_t* mark) {
    // decode only the pages any tile selected (mark bitmap built by
    // mark_selected_pages); skips the untouched majority of the cache.
    // Grid (pages, 2) x 256 threads: each 128-thread half owns one
    // (page, head) pair - half the CTAs of the (pages,4)x128 layout, which
    // matters because the kernel is CTA-cycling-bound at small selections.
    // PDL: launched with programmatic stream serialization, so this grid
    // may run while mark_selected_pages still executes. The mark is
    // per-(page, head): only this half reads and re-zeros its own byte.
    // Marks are monotonic 0->1 within a run; an L2 read of 1 is therefore
    // always a real selection, a 0 read must be confirmed after
    // grid_dep_wait. All mark reads bypass L1 because a line cached before
    // the wait could be stale.
    const int page=blockIdx.x;
    const int head=blockIdx.y;
    const int local=threadIdx.x;
    uint32_t* mymark=reinterpret_cast<uint32_t*>(mark)+int64_t(page)*4+head;
    const bool wide=p.batches*((p.max_q+7)/8)>g_wide_thresh;
    if(!__ldcg(mymark)) {
        grid_dep_wait();
        if(!__ldcg(mymark)) return;
    }
    // Overlap path only (narrow grids): read the epoch AFTER a wait so the
    // stamp below can never carry a stale epoch (the bump is a plain store
    // in the mark grid). The wide path skips the wait/stamp entirely.
    uint32_t* stamps=nullptr; unsigned e=0;
    if(!wide) {
        grid_dep_wait();
        uint32_t* epoch=reinterpret_cast<uint32_t*>(mark+size_t(p.pages)*16);
        stamps=epoch+1;
        e=__ldcg(epoch);
        grid_dep_launch();
    }
    // Latency-bound by design: 18KB per (page,head) from cold DRAM is far
    // below bandwidth limits, so stage all 8 load rounds into registers
    // first - 16 independent 64-bit loads in flight instead of 8 dependent
    // load->decode->store chains.
    uint64_t pk[8],pv[8]; uint8_t sk[8],sv[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
        int i=local+j*128;
        int token=i/8,g=i%8,d=g*16;
        int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
        int64_t so=int64_t(page)*4096+head*1024+sf;
        pk[j]=*reinterpret_cast<const uint64_t*>(p.k+int64_t(page)*p.kp+head*8192+token*64+d/2);
        pv[j]=*reinterpret_cast<const uint64_t*>(p.v+int64_t(page)*p.vp+head*8192+token*64+d/2);
        sk[j]=p.ks[so];sv[j]=p.vs[so];
    }
    #pragma unroll
    for(int j=0;j<8;++j) {
        int i=local+j*128;
        int token=i/8,g=i%8,d=g*16;
        uint4 key,value;
        #pragma unroll
        for(int c=0;c<4;++c) {
            reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(pk[j]>>(c*16)),sk[j]);
            reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(pv[j]>>(c*16)),sv[j]);
        }
        int64_t offset=(int64_t(page)*4+head)*16384+token*128+d;
        *reinterpret_cast<uint4*>(p.k8+offset)=key;
        *reinterpret_cast<uint4*>(p.v8+offset)=value;
    }
    // Publish this (page,head) ONLY on the overlap path (narrow grids):
    // wide grids never launch dependents early, so the stamp + its
    // device-wide __threadfence would be pure overhead there (r52/r53:
    // ~512 fences cost 1.3pt at q128/b8).
    if(!wide) {
        __threadfence();
        if(local==0)
            *reinterpret_cast<volatile uint32_t*>(stamps+int64_t(page)*4+head)=e;
    } else {
        grid_dep_launch();  // wide: serial semantics, trigger at the tail
    }
    // Self-clean: this half alone owns mymark, so the next launch (CUDA
    // Graph replay included) finds it zeroed - no separate memset node.
    if(local==0) *mymark=0;}

__global__ void decode_working_cache_list(Params p, uint8_t* mark) {
    // List-driven variant of decode_working_cache for narrow grids whose
    // selection covers a small fraction of the page pool: the grid is the
    // compacted (page,head) list built by mark_selected_pages_list instead
    // of every page, removing the CTA-cycling sweep over unselected pages
    // (the full-page grid is dispatch-throughput bound at ~1ns/CTA, which
    // dominates the kernel once selection covers less than half the pool).
    // Every CTA reads its slot only AFTER grid_dep_wait: all appends
    // precede their marker thread's trigger, so the post-wait read observes
    // the complete append set of this run - lossless by construction, and
    // immune to the stale-slot/pre-wait-read pattern a compacted list would
    // otherwise have. The slot is reset to -1 BEFORE the stamp publishes
    // the page, so resident consumption of the stamp implies the slot was
    // already cleaned - no cross-replay slot race.
    const int slot=blockIdx.x;
    const int local=threadIdx.x;
    uint32_t* epoch=reinterpret_cast<uint32_t*>(mark+size_t(p.pages)*16);
    int32_t* list=reinterpret_cast<int32_t*>(mark+size_t(p.pages)*32+4);
    grid_dep_wait();
    const int e=__ldcg(list+slot);
    if(e<0) return;
    const int page=e>>2;
    const int head=e&3;
    if(int64_t(page)>=p.pages) return;
    const bool wide=p.batches*((p.max_q+7)/8)>g_wide_thresh;
    uint32_t* stamps=nullptr; unsigned ep=0;
    if(!wide) {
        ep=__ldcg(epoch);
        stamps=epoch+1;
        grid_dep_launch();
    }
    uint64_t pk[8],pv[8]; uint8_t sk[8],sv[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
        int i=local+j*128;
        int token=i/8,g=i%8,d=g*16;
        int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
        int64_t so=int64_t(page)*4096+head*1024+sf;
        pk[j]=*reinterpret_cast<const uint64_t*>(p.k+int64_t(page)*p.kp+head*8192+token*64+d/2);
        pv[j]=*reinterpret_cast<const uint64_t*>(p.v+int64_t(page)*p.vp+head*8192+token*64+d/2);
        sk[j]=p.ks[so];sv[j]=p.vs[so];
    }
    #pragma unroll
    for(int j=0;j<8;++j) {
        int i=local+j*128;
        int token=i/8,g=i%8,d=g*16;
        uint4 key,value;
        #pragma unroll
        for(int c=0;c<4;++c) {
            reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(pk[j]>>(c*16)),sk[j]);
            reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(pv[j]>>(c*16)),sv[j]);
        }
        int64_t offset=(int64_t(page)*4+head)*16384+token*128+d;
        *reinterpret_cast<uint4*>(p.k8+offset)=key;
        *reinterpret_cast<uint4*>(p.v8+offset)=value;
    }
    if(!wide) {
        if(local==0) list[slot]=-1;
        __threadfence();
        if(local==0)
            *reinterpret_cast<volatile uint32_t*>(stamps+int64_t(page)*4+head)=ep;
    } else {
        if(local==0) list[slot]=-1;
        grid_dep_launch();
    }
    if(local==0) reinterpret_cast<uint32_t*>(mark)[int64_t(page)*4+head]=0;
}

template<size_t... I>
__device__ __forceinline__ void load32_impl(uint32_t addr,float* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_32dp32b32x::copy(addr,reinterpret_cast<uint32_t*>(v)[I]...);
}
__device__ __forceinline__ void load32(uint32_t addr,float* v) {
    load32_impl(addr,v,std::make_index_sequence<32>{});
}
template<size_t... I>
__device__ __forceinline__ void load16_impl(uint32_t addr,float* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_32dp32b16x::copy(addr,reinterpret_cast<uint32_t*>(v)[I]...);
}
__device__ __forceinline__ void load16(uint32_t addr,float* v) {
    load16_impl(addr,v,std::make_index_sequence<16>{});
}
__device__ __forceinline__ void load32max(uint32_t addr,float* v,float& maximum) {
    asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, [%33];"
        : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
          "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15]),
          "=f"(v[16]),"=f"(v[17]),"=f"(v[18]),"=f"(v[19]),"=f"(v[20]),"=f"(v[21]),"=f"(v[22]),"=f"(v[23]),
          "=f"(v[24]),"=f"(v[25]),"=f"(v[26]),"=f"(v[27]),"=f"(v[28]),"=f"(v[29]),"=f"(v[30]),"=f"(v[31]),"=f"(maximum)
        : "r"(addr));
}
template<size_t... I>
__device__ __forceinline__ void store32_impl(uint32_t addr,const uint32_t* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_STORE_32dp32b32x::copy(v[I]...,addr);
}
__device__ __forceinline__ void store32(uint32_t addr,const void* v) {
    store32_impl(addr,static_cast<const uint32_t*>(v),std::make_index_sequence<32>{});
}
template<size_t... I>
__device__ __forceinline__ void store16_impl(uint32_t addr,const uint32_t* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_STORE_32dp32b16x::copy(v[I]...,addr);
}
__device__ __forceinline__ void store16(uint32_t addr,const void* v) {
    store16_impl(addr,static_cast<const uint32_t*>(v),std::make_index_sequence<16>{});
}
__device__ __forceinline__ uint64_t descriptor(uint8_t* ptr) {
    return cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(
        cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(ptr)),Layout{}));
}
__device__ __forceinline__ void after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__global__ __launch_bounds__(128) void grouped_cutlass(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<Shared*>(smem);
    const int tid=threadIdx.x, warp=tid/32;
    const int batch=blockIdx.y/4, head=blockIdx.y%4;
    const int start=p.cuq[batch], stop=p.cuq[batch+1];
    const int qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    const int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) cutlass::arch::ClusterBarrier::init(&s.done,1);
    if(warp==0) {
        cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    for(int i=tid;i<4096;i+=128) {
        int row=i/32,d=i%32*4,q=qbase+row/16;
        uint32_t x=q<stop?*reinterpret_cast<const uint32_t*>(p.q+int64_t(q)*p.qn+(head*16+row%16)*p.qh+d):0;
        *reinterpret_cast<uint32_t*>(s.q+Layout{}(cute::make_coord(row,d)))=x;
    }
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    const uint32_t t=s.tmem, row_addr=(warp*32)<<16;
    const uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
    const uint64_t dq=descriptor(s.q),dk=descriptor(s.k),dv=descriptor(s.v);
    float zeros[32]={};
    #pragma unroll
    for(int c=0;c<128;c+=32) store32(t+256+row_addr+c,zeros);
    cutlass::arch::fence_view_async_tmem_store();
    float m=-CUDART_INF_F,z=0.f;
    const int query=qbase+tid/16;
    const int causal_limit=query-start+length-(stop-start);
    for(int slot=0;slot<16;++slot) {
        int block=p.topk[head*p.th+int64_t(qbase)*p.tq+slot];
        bool valid=block>=0&&block<p.max_pages&&block*128<length;
        int page=valid?p.table[batch*p.tb+block]:-1;
        valid=valid&&page>=0&&page<p.pages;
        if(!valid) continue;
        for(int i=tid;i<4096;i+=128) {
            int token=i/32,d=i%32*4,g=d/16;
            int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
            int64_t po=int64_t(page)*p.kp+head*8192+token*64+d/2;
            int64_t vo=int64_t(page)*p.vp+head*8192+token*64+d/2;
            int64_t so=int64_t(page)*4096+head*1024+sf;
            uint32_t k=decode4(*reinterpret_cast<const uint16_t*>(p.k+po),p.ks[so]);
            uint32_t v=decode4(*reinterpret_cast<const uint16_t*>(p.v+vo),p.vs[so]);
            if(block*128+token>=length) k=v=0;
            *reinterpret_cast<uint32_t*>(s.k+Layout{}(cute::make_coord(token,d)))=k;
            #pragma unroll
            for(int j=0;j<4;++j) s.v[Layout{}(cute::make_coord(d+j,token))]=uint8_t(v>>(j*8));
        }
        cutlass::arch::fence_view_async_shared();
        before_sync(); __syncthreads(); after_sync();
        if(warp==0) {
            #pragma unroll
            for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
            cutlass::arch::umma_arrive(&s.done);
        }
        cutlass::arch::ClusterBarrier::wait(&s.done,0);
        after_sync();
        float scores[128];
        #pragma unroll
        for(int c=0;c<128;c+=32) load32(t+row_addr+c,scores+c);
        cutlass::arch::fence_view_async_tmem_load();
        float maximum=-CUDART_INF_F;
        #pragma unroll
        for(int c=0;c<128;++c) {
            if(query>=stop||block*128+c>=length||block*128+c>causal_limit) scores[c]=-CUDART_INF_F;
            maximum=fmaxf(maximum,scores[c]);
        }
        float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
        #pragma unroll
        for(int c=0;c<128;++c) {
            float e=fmaf(scores[c],p.alpha,shift);
            scores[c]=(c>=32&&c<96&&c%16>=12)?exp_emulated(e):exp_native(e);
        }
        float sums[8]={};
        #pragma unroll
        for(int j=0;j<16;++j) {
            #pragma unroll
            for(int c=0;c<8;++c) sums[c]=sums[c]+scores[j*8+c];
        }
        float den=((sums[0]+sums[2])+(sums[4]+sums[6]))+((sums[1]+sums[3])+(sums[5]+sums[7]));
        float inv=rcp_native(den>0.f?den:1.f);
        float lp=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
        uint32_t packed[32];
        #pragma unroll
        for(int c=0;c<32;++c) {
            uint32_t v=0;
            #pragma unroll
            for(int j=0;j<4;++j) v|=uint32_t(F8(scores[c*4+j]).raw())<<(j*8);
            packed[c]=v;
        }
        store32(t+64+row_addr,packed);
        cutlass::arch::fence_view_async_tmem_store();
        before_sync(); __syncthreads(); after_sync();
        if(warp==0) {
            #pragma unroll
            for(int k=0;k<128;k+=32) PVMma::fma(t+64+k/4,dv+k/16,t+128,k!=0,inst);
            cutlass::arch::umma_arrive(&s.done);
        }
        cutlass::arch::ClusterBarrier::wait(&s.done,1);
        after_sync();
        float nm=fmaxf(m,lp),sm=nm==-CUDART_INF_F?0.f:nm;
        float a=z>0.f?exp_native(m-sm):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-sm):0.f;
        #pragma unroll
        for(int c=0;c<128;c+=32) {
            float page_o[32],running[32];
            load32(t+128+row_addr+c,page_o); load32(t+256+row_addr+c,running);
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for(int j=0;j<32;++j) {
                float part=__bfloat162float(__float2bfloat16_rn(page_o[j]*inv));
                running[j]=running[j]*a+part*b;
            }
            store32(t+256+row_addr+c,running);
        }
        cutlass::arch::fence_view_async_tmem_store();
        m=nm; z=z*a+b;
    }
    float inv=rcp_native(z>0.f?z:1.f);
    #pragma unroll
    for(int c=0;c<128;c+=32) {
        float running[32]; load32(t+256+row_addr+c,running);
        cutlass::arch::fence_view_async_tmem_load();
        if(query<stop) {
            #pragma unroll
            for(int j=0;j<32;++j) p.out[(int64_t(query)*64+head*16+tid%16)*128+c+j]=__float2bfloat16_rn(running[j]*inv);
        }
    }
    if(query<stop) p.lse[int64_t(query)*64+head*16+tid%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
    before_sync(); __syncthreads();
    if(warp==0) cute::TMEM::Allocator1Sm().free(t,512);
}
struct alignas(128) PipelineShared {
    alignas(128) uint8_t q[16384], k[2][16384], v[2][16384];
#if MSA_DUAL
    alignas(128) uint8_t packed_k[8192],packed_v[8192],scale_k[1024],scale_v[1024];
#endif
    float inv[2][128], lp[2][128];
    alignas(8) uint64_t kv_ready[2], kv_free[2], s_ready[2], s_free[2], p_half[2], p_ready[2], stats_free[2];
    uint64_t o_ready, o_free, done;
    uint32_t tmem;
};
using OutputLayout = decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                                                 cute::Shape<cute::_128,cute::_128>{}));

template<int Id>
__device__ __forceinline__ void wg_sync() {
    asm volatile("bar.sync %0, 128;" :: "n"(Id) : "memory");
}
__device__ __forceinline__ void loader_sync() {
    asm volatile("bar.sync 4, 96;" ::: "memory");
}

template<bool SharedP=false>
__device__ __forceinline__ void page_softmax(uint32_t score,uint32_t row_addr,int block,int limit,
                                          int length,bool valid,float alpha,float& inv,float& lp,uint64_t* debug,uint64_t* half_ready,uint8_t* shared_p=nullptr,uint64_t* score_free=nullptr,uint64_t* half2_ready=nullptr) {
    float v[128],tile_max[4];
    #pragma unroll
    for(int c=0;c<128;c+=32) load32max(score+row_addr+c,v+c,tile_max[c/32]);
    cutlass::arch::fence_view_async_tmem_load();
    if(score_free&&threadIdx.x%128==0) cutlass::arch::ClusterBarrier::arrive(score_free);
    if(debug&&threadIdx.x%128==0) debug[4]=clock64();
    const int last=valid?min(length-1,limit)-block*128:-1;
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
    if(debug&&threadIdx.x%128==0) debug[5]=clock64();
    float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-alpha);
    if(half2_ready) {
        // 32-col early arrival: pack cols 0-31, signal, then cols 32-63.
        #pragma unroll
        for(int c=0;c<32;c+=2) {
            affine_pair(v[c],v[c+1],alpha,shift);
            v[c]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c]):exp_native(v[c]);
            v[c+1]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c+1]):exp_native(v[c+1]);
        }
        uint32_t packed_q0[8];
        #pragma unroll
        for(int c=0;c<8;++c) packed_q0[c]=pack_probability(v[c*4],v[c*4+1],v[c*4+2],v[c*4+3]);
        #pragma unroll
        for(int c=0;c<8;c+=4) *reinterpret_cast<uint4*>(shared_p+Layout{}(cute::make_coord(threadIdx.x%128,c*4)))=*reinterpret_cast<uint4*>(packed_q0+c);
        cutlass::arch::fence_view_async_shared();
        before_sync();
        if(threadIdx.x/128==0) wg_sync<1>(); else wg_sync<2>();
        if(threadIdx.x%128==0) cutlass::arch::ClusterBarrier::arrive(half_ready);
        #pragma unroll
        for(int c=32;c<64;c+=2) {
            affine_pair(v[c],v[c+1],alpha,shift);
            v[c]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c]):exp_native(v[c]);
            v[c+1]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c+1]):exp_native(v[c+1]);
        }
        uint32_t packed_q1[8];
        #pragma unroll
        for(int c=0;c<8;++c) packed_q1[c]=pack_probability(v[32+c*4],v[32+c*4+1],v[32+c*4+2],v[32+c*4+3]);
        #pragma unroll
        for(int c=0;c<8;c+=4) *reinterpret_cast<uint4*>(shared_p+Layout{}(cute::make_coord(threadIdx.x%128,32+c*4)))=*reinterpret_cast<uint4*>(packed_q1+c);
        cutlass::arch::fence_view_async_shared();
        before_sync();
        if(threadIdx.x/128==0) wg_sync<1>(); else wg_sync<2>();
        if(threadIdx.x%128==0) cutlass::arch::ClusterBarrier::arrive(half2_ready);
    } else {
    #pragma unroll
    for(int c=0;c<64;c+=2) {
        affine_pair(v[c],v[c+1],alpha,shift);
        v[c]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c]):exp_native(v[c]);
        v[c+1]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c+1]):exp_native(v[c+1]);
    }
    uint32_t packed_first[16];
    #pragma unroll
    for(int c=0;c<16;++c) packed_first[c]=pack_probability(v[c*4],v[c*4+1],v[c*4+2],v[c*4+3]);
    if constexpr(SharedP) {
        #pragma unroll
        for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(shared_p+Layout{}(cute::make_coord(threadIdx.x%128,c*4)))=*reinterpret_cast<uint4*>(packed_first+c);
        cutlass::arch::fence_view_async_shared();
    } else {
        store16(score+64+row_addr,packed_first);
        cutlass::arch::fence_view_async_tmem_store();
    }
    before_sync();
    if(threadIdx.x/128==0) wg_sync<1>(); else wg_sync<2>();
    if(threadIdx.x%128==0) cutlass::arch::ClusterBarrier::arrive(half_ready);
    }
    #pragma unroll
    for(int c=64;c<128;c+=2) {
        affine_pair(v[c],v[c+1],alpha,shift);
        v[c]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c]):exp_native(v[c]);
        v[c+1]=(c>=32&&c<96&&c%16>=12)?exp_emulated(v[c+1]):exp_native(v[c+1]);
    }
    if(debug&&threadIdx.x%128==0) debug[6]=clock64();
    float sums[8]={};
    #pragma unroll
    for(int j=0;j<16;++j) {
        #pragma unroll
        for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[j*8+c],v[j*8+c+1]);
    }
    float den=((sums[0]+sums[2])+(sums[4]+sums[6]))+((sums[1]+sums[3])+(sums[5]+sums[7]));
    inv=rcp_native(den>0.f?den:1.f);
    lp=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,alpha,log_native(den));
    if(debug&&threadIdx.x%128==0) debug[7]=clock64();
    uint32_t packed[16];
    #pragma unroll
    for(int c=0;c<16;++c) {
        packed[c]=pack_probability(v[64+c*4],v[64+c*4+1],v[64+c*4+2],v[64+c*4+3]);
    }
    if(debug&&threadIdx.x%128==0) debug[8]=clock64();
    if constexpr(SharedP) {
        #pragma unroll
        for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(shared_p+Layout{}(cute::make_coord(threadIdx.x%128,64+c*4)))=*reinterpret_cast<uint4*>(packed+c);
        cutlass::arch::fence_view_async_shared();
    } else {
        store16(score+80+row_addr,packed);
        cutlass::arch::fence_view_async_tmem_store();
    }
    if(debug&&threadIdx.x%128==0) debug[9]=clock64();
}

__global__ __launch_bounds__(512,1) void grouped_cutlass_pipeline(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<PipelineShared*>(smem);
    int tid=threadIdx.x, wg=tid/128, local=tid%128, warp=local/32;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1];
    int qbase=start+int(blockIdx.x)*8;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            cutlass::arch::ClusterBarrier::init(&s.kv_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.kv_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.s_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.s_free[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_ready[i],1);
            cutlass::arch::ClusterBarrier::init(&s.p_half[i],1);
            cutlass::arch::ClusterBarrier::init(&s.stats_free[i],1);
        }
        cutlass::arch::ClusterBarrier::init(&s.o_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.o_free,1);
        cutlass::arch::ClusterBarrier::init(&s.done,1);
    }
    if(tid<32) {
        cute::TMEM::Allocator1Sm().allocate(512,&s.tmem);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    for(int i=tid;i<4096;i+=512) {
        int row=i/32,d=i%32*4,q=qbase+row/16;
        uint32_t x=q<stop?*reinterpret_cast<const uint32_t*>(p.q+int64_t(q)*p.qn+(head*16+row%16)*p.qh+d):0;
        *reinterpret_cast<uint32_t*>(s.q+Layout{}(cute::make_coord(row,d)))=x;
    }
    cutlass::arch::fence_barrier_init();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    uint32_t t=s.tmem,row_addr=(warp*32)<<16;
    uint64_t* debug=MSA_TRACE&&p.debug?p.debug+(int(blockIdx.y)*gridDim.x+blockIdx.x)*576+(MSA_DUAL?max(0,wg-1):wg)*192:nullptr;
    if(wg<(MSA_DUAL?2:1)) {
        cutlass::arch::warpgroup_reg_alloc<MSA_DUAL&&!MSA_DECODED?176:192>();
        int query=qbase+local/16,limit=query-start+length-(stop-start);
        for(int j=MSA_DUAL?wg:0;j<16;j+=MSA_DUAL?2:1) {
            int slot=j%2;
            if(debug&&local==0) debug[j*12]=clock64();
            cutlass::arch::ClusterBarrier::wait(&s.s_ready[slot],(j/2)%2);
            after_sync();
            if(debug&&local==0) debug[j*12+1]=clock64();
            int block=p.topk[head*p.th+int64_t(qbase)*p.tq+j];
            bool valid=block>=0&&block<p.max_pages&&block*128<length;
            int page=valid?p.table[batch*p.tb+block]:-1;
            valid=valid&&page>=0&&page<p.pages&&query<stop;
            float inv,lp;
            page_softmax(t+slot*128,row_addr,block,limit,length,valid,p.alpha,inv,lp,debug?debug+j*12:nullptr,&s.p_half[slot]);
            if(j>=2) cutlass::arch::ClusterBarrier::wait(&s.stats_free[slot],((j/2)-1)%2);
            s.inv[slot][local]=inv;s.lp[slot][local]=lp;
            if(debug&&local==0) debug[j*12+2]=clock64();
            before_sync();
            if(wg==0) wg_sync<1>(); else wg_sync<2>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready[slot]);
            if(debug&&local==0) debug[j*12+3]=clock64();
        }
    } else if(wg==(MSA_DUAL?2:1)) {
        cutlass::arch::warpgroup_reg_dealloc<MSA_DUAL?96:112>();
        float zero[32]={};
        #pragma unroll
        for(int c=0;c<128;c+=32) store32(t+384+row_addr+c,zero);
        cutlass::arch::fence_view_async_tmem_store();
        float m=-CUDART_INF_F,z=0.f;
        for(int j=0;j<16;++j) {
            int slot=j%2;
            if(debug&&local==0) debug[j*12]=clock64();
            cutlass::arch::ClusterBarrier::wait(&s.o_ready,j%2);
            after_sync();
            if(debug&&local==0) debug[j*12+1]=clock64();
            float inv=s.inv[slot][local],lp=s.lp[slot][local];
            wg_sync<3>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.stats_free[slot]);
            float nm=fmaxf(m,lp),safe=nm==-CUDART_INF_F?0.f:nm;
            float a=z>0.f?exp_native(m-safe):0.f,b=lp!=-CUDART_INF_F?exp_native(lp-safe):0.f;
            #pragma unroll
            for(int c=0;c<128;c+=16) {
                float page_o[16],running[16];
                load16(t+256+row_addr+c,page_o);load16(t+384+row_addr+c,running);
                cutlass::arch::fence_view_async_tmem_load();
                #pragma unroll
                for(int k=0;k<16;k+=2) merge_pair(running[k],running[k+1],page_o[k],page_o[k+1],inv,a,b);
                store16(t+384+row_addr+c,running);
            }
            cutlass::arch::fence_view_async_tmem_store();
            m=nm;z=z*a+b;
            if(debug&&local==0) debug[j*12+2]=clock64();
            before_sync();wg_sync<3>();
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.o_free);
            if(debug&&local==0) debug[j*12+3]=clock64();
        }
        float inv=rcp_native(z>0.f?z:1.f);
        auto* output=reinterpret_cast<__nv_bfloat16*>(s.k);
        #pragma unroll
        for(int c=0;c<128;c+=32) {
            float running[32];load32(t+384+row_addr+c,running);
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for(int k=0;k<32;k+=2) {
                __nv_bfloat162 pair=__floats2bfloat162_rn(running[k]*inv,running[k+1]*inv);
                *reinterpret_cast<__nv_bfloat162*>(output+OutputLayout{}(cute::make_coord(local,c+k)))=pair;
            }
        }
        int query=qbase+local/16;
        if(query<stop) p.lse[int64_t(query)*64+head*16+local%16]=z>0.f?(m+log_native(z))*0.6931471805599453f:-CUDART_INF_F;
        wg_sync<3>();
        if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
    } else {
        if constexpr(MSA_DUAL) cutlass::arch::warpgroup_reg_dealloc<MSA_DECODED?32:64>();
        else if(wg==2) cutlass::arch::warpgroup_reg_dealloc<128>();
        else cutlass::arch::warpgroup_reg_dealloc<80>();
        if(MSA_DUAL?local<96:wg==2) {
        for(int j=0;j<16;++j) {
            int slot=j%2;
            if(debug&&local==0) debug[j*12]=clock64();
            if(j>=2) cutlass::arch::ClusterBarrier::wait(&s.kv_free[slot],((j/2)-1)%2);
            if(debug&&local==0) debug[j*12+1]=clock64();
            int block=p.topk[head*p.th+int64_t(qbase)*p.tq+j];
            bool valid=block>=0&&block<p.max_pages&&block*128<length;
            int page=valid?p.table[batch*p.tb+block]:-1;
            valid=valid&&page>=0&&page<p.pages;
#if MSA_DUAL
            const int safe_page=valid?page:0;
            using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
            if(MSA_DECODED||p.k8) {
                for(int i=local;i<1024;i+=96) {
                    int token=i/8,d=i%8*16;
                    int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                    bool present=valid&&block*128+token<length;
                    Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.k[slot]+Layout{}(cute::make_coord(token,d))),present);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.v[slot]+VLayout{}(cute::make_coord(d,token))),present);
                }
                cute::cp_async_fence();cute::cp_async_wait<0>();
            } else {
            for(int i=local;i<512;i+=96) {
                Copy::copy(*reinterpret_cast<const uint4*>(p.k+int64_t(safe_page)*p.kp+head*8192+i*16),
                           *reinterpret_cast<uint4*>(s.packed_k+i*16),valid);
                Copy::copy(*reinterpret_cast<const uint4*>(p.v+int64_t(safe_page)*p.vp+head*8192+i*16),
                           *reinterpret_cast<uint4*>(s.packed_v+i*16),valid);
            }
            if(local<64) {
                Copy::copy(*reinterpret_cast<const uint4*>(p.ks+int64_t(safe_page)*4096+head*1024+local*16),
                           *reinterpret_cast<uint4*>(s.scale_k+local*16),valid);
                Copy::copy(*reinterpret_cast<const uint4*>(p.vs+int64_t(safe_page)*4096+head*1024+local*16),
                           *reinterpret_cast<uint4*>(s.scale_v+local*16),valid);
            }
            cute::cp_async_fence();cute::cp_async_wait<0>();loader_sync();
            for(int i=local;i<1024;i+=96) {
                int token=i/8,g=i%8,d=g*16;
                int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
                uint64_t pk=*reinterpret_cast<const uint64_t*>(s.packed_k+token*64+d/2);
                uint64_t pv=*reinterpret_cast<const uint64_t*>(s.packed_v+token*64+d/2);
                uint8_t sk=s.scale_k[sf],sv=s.scale_v[sf];
                uint4 key,value;
                #pragma unroll
                for(int c=0;c<4;++c) {
                    reinterpret_cast<uint32_t*>(&key)[c]=block*128+token<length?decode4(uint16_t(pk>>(c*16)),sk):0;
                    reinterpret_cast<uint32_t*>(&value)[c]=block*128+token<length?decode4(uint16_t(pv>>(c*16)),sv):0;
                }
                *reinterpret_cast<uint4*>(s.k[slot]+Layout{}(cute::make_coord(token,d)))=key;
                *reinterpret_cast<uint4*>(s.v[slot]+VLayout{}(cute::make_coord(d,token)))=value;
            }
            }
            if(debug&&local==0) debug[j*12+2]=clock64();
            cutlass::arch::fence_view_async_shared();loader_sync();
#else
            if(MSA_DECODED||p.k8) {
                const int safe_page=valid?page:0;
                using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
                #pragma unroll
                for(int i=0;i<8;++i) {
                    int token=local/8+i*16,d=local%8*16;
                    int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                    bool present=valid&&block*128+token<length;
                    Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.k[slot]+Layout{}(cute::make_coord(token,d))),present);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.v[slot]+VLayout{}(cute::make_coord(d,token))),present);
                }
                cute::cp_async_fence();cute::cp_async_wait<0>();
            } else {
            uint64_t packed_k[8],packed_v[8];
            uint8_t scale_k[8],scale_v[8];
            const int g=local%8,d=g*16;
            #pragma unroll
            for(int i=0;i<8;++i) {
                int token=local/8+i*16;
                packed_k[i]=packed_v[i]=0;scale_k[i]=scale_v[i]=0;
                if(valid&&block*128+token<length) {
                    int sf=(g/4)*512+(token%32)*16+(token/32)*4+g%4;
                    int64_t po=int64_t(page)*p.kp+head*8192+token*64+d/2;
                    int64_t vo=int64_t(page)*p.vp+head*8192+token*64+d/2;
                    int64_t so=int64_t(page)*4096+head*1024+sf;
                    packed_k[i]=*reinterpret_cast<const uint64_t*>(p.k+po);
                    packed_v[i]=*reinterpret_cast<const uint64_t*>(p.v+vo);
                    scale_k[i]=p.ks[so];scale_v[i]=p.vs[so];
                }
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for(int i=0;i<8;++i) {
                int token=local/8+i*16;
                uint4 key,value;
                #pragma unroll
                for(int c=0;c<4;++c) {
                    reinterpret_cast<uint32_t*>(&key)[c]=decode4(uint16_t(packed_k[i]>>(c*16)),scale_k[i]);
                    reinterpret_cast<uint32_t*>(&value)[c]=decode4(uint16_t(packed_v[i]>>(c*16)),scale_v[i]);
                }
                *reinterpret_cast<uint4*>(s.k[slot]+Layout{}(cute::make_coord(token,d)))=key;
                *reinterpret_cast<uint4*>(s.v[slot]+VLayout{}(cute::make_coord(d,token)))=value;
            }
            }
            if(debug&&local==0) debug[j*12+2]=clock64();
            cutlass::arch::fence_view_async_shared();wg_sync<4>();
#endif
            if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready[slot]);
            if(debug&&local==0) debug[j*12+3]=clock64();
        }
        } else if(MSA_DUAL||warp==0) {
            uint64_t dq=descriptor(s.q);
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,128,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            for(int j=0;j<17;++j) {
                if(j<16) {
                    int slot=j%2;
                    if(j>=2) cutlass::arch::ClusterBarrier::wait(&s.s_free[slot],((j/2)-1)%2);
                    cutlass::arch::ClusterBarrier::wait(&s.kv_ready[slot],(j/2)%2);
                    after_sync();
                    uint64_t dk=descriptor(s.k[slot]);
                    #pragma unroll
                    for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t+slot*128,k!=0,inst);
                    cutlass::arch::umma_arrive(&s.s_ready[slot]);
                }
                if(j>0) {
                    int page=j-1,slot=page%2;
                    cutlass::arch::ClusterBarrier::wait(&s.p_half[slot],(page/2)%2);
                    if(page>0) cutlass::arch::ClusterBarrier::wait(&s.o_free,(page-1)%2);
                    after_sync();
                    uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(
                        cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.v[slot])),VLayout{}));
                    #pragma unroll
                    for(int k=0;k<64;k+=32) PVMmaMN::fma(t+slot*128+64+k/4,dv+VLayout{}(cute::make_coord(0,k))/16,t+256,k!=0,inst_pv);
                    cutlass::arch::ClusterBarrier::wait(&s.p_ready[slot],(page/2)%2);
                    after_sync();
                    #pragma unroll
                    for(int k=64;k<128;k+=32) PVMmaMN::fma(t+slot*128+64+k/4,dv+VLayout{}(cute::make_coord(0,k))/16,t+256,true,inst_pv);
                    cutlass::arch::umma_arrive(&s.o_ready);
                    cutlass::arch::umma_arrive(&s.s_free[slot]);
                    cutlass::arch::umma_arrive(&s.kv_free[slot]);
                }
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.k);
    for(int i=tid;i<2048;i+=512) {
        int row=i/16,d=i%16*8,query=qbase+row/16;
        if(query<stop) {
            uint4 value=*reinterpret_cast<uint4*>(output+OutputLayout{}(cute::make_coord(row,d)));
            *reinterpret_cast<uint4*>(p.out+(int64_t(query)*64+head*16+row%16)*128+d)=value;
        }
    }
    before_sync();__syncthreads();
    if(tid<32) cute::TMEM::Allocator1Sm().free(t,512);
}

} // namespace rtp_llm

#include "late_merge.cuh"
#include "compact_late.cuh"
#include "dual_query.cuh"
#include "split_merge.cuh"
#include "resident.cuh"
#include "resident2.cuh"
#include "resident3.cuh"
#include "resident4.cuh"
#include "resident5.cuh"
#include "resident6.cuh"
#include "resident7.cuh"
#include "resident8.cuh"
#include "resident9.cuh"
#include "resident11.cuh"
#include "resident12.cuh"
#include "resident12g2.cuh"
#include "resident12p.cuh"
#include "resident13.cuh"
#include "decode.cuh"
#include "decode_partial.cuh"
#include "decode_m64.cuh"
#include "decode_warp.cuh"
#include "decode_pair.cuh"
#include "m64_probe.cuh"
#include "mma_probe.cuh"

static int g_selfmark=0;
static int g_coop=0;
extern "C" void msa_set_selfmark(int v) { g_selfmark=v; }
extern "C" void msa_set_coop(int v) { g_coop=v; }
static int g_g2_groups=2;
static int g_mark1=0;
static int g_listcache=1;
extern "C" void msa_set_listcache(int v) { g_listcache=v; }
extern "C" void msa_set_mark1(int v) { g_mark1=v; }
// Host mirror of the device g_wide_thresh: host code must not read the
// __device__ variable directly (its host-side shadow reads 0, which
// silently forced flags=null at every shape and disabled the loader's
// stamp spins - the early narrow-path PDL trigger then left a
// read-before-write race on k8/v8 that frozen fixtures masked).
static int g_wide_thresh_host=140;
extern "C" void msa_set_wide_thresh(int v) {
    g_wide_thresh_host=v;
    cudaMemcpyToSymbol(rtp_llm::g_wide_thresh,&v,sizeof(int));
}
extern "C" void msa_set_g2_groups(int v) { g_g2_groups=v; }

// Launch a kernel with cudaLaunchAttributeProgrammaticStreamSerialization:
// the grid may launch while the previous kernel in the stream still runs
// (stream capture / CUDA Graph replay preserves the attribute). The kernel
// itself guards dependent reads with grid_dep_wait.
template<class K, class... A>
static cudaError_t launch_with_pdl(K kernel, dim3 grid, dim3 block, size_t smem,
                                   cudaStream_t stream, A&&... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim=grid;
    cfg.blockDim=block;
    cfg.dynamicSmemBytes=smem;
    cfg.stream=stream;
    cudaLaunchAttribute attr{};
    attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;
    cfg.attrs=&attr;
    cfg.numAttrs=1;
    return cudaLaunchKernelEx(&cfg,kernel,std::forward<A>(args)...);
}

extern "C" int msa_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    const uint32_t* flags=nullptr;  // [epoch][stamps] after the mark bitmap
#if MSA_RESIDENT
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::ResidentShared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT2
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident2,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident2Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident2,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT3
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident3,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident3Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident3,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT4
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident4,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident4Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident4,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT5
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident5,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident5Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident5,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT6
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident6,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident6Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident6,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT7
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident7,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident7Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident7,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT8
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident8,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident8Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident8,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT11
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident11,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident11Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident11,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT12
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident12,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident12Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident12,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_RESIDENT13
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident13,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident13Shared));
    if(err==cudaSuccess) err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident13,cudaFuncAttributePreferredSharedMemoryCarveout,100);
#elif MSA_SPLIT_MERGE
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_split_merge,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::SplitMergeShared));
#elif MSA_DUAL_QUERY
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_dual_query,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DualQueryShared));
#elif MSA_COMPACT
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_compact,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::CompactShared));
#elif MSA_LATE_MERGE
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_late,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::LateShared));
#else
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_pipeline,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::PipelineShared));
#endif
    if(err!=cudaSuccess) return int(err);
    if(p->k8) {
        if(g_coop) {
            static uint8_t* cmark=nullptr;
            static int cmark_pages=0;
            if(!cmark||cmark_pages<p->pages) {
                if(cmark) cudaFree(cmark);
                cudaMalloc(&cmark,size_t(p->pages)*16);
                cudaMemsetAsync(cmark,0,size_t(p->pages)*16,stream);
                cmark_pages=p->pages;
            }
            int dev=0; cudaGetDevice(&dev);
            int max_blocks=0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,rtp_llm::decode_cache_coop,128,0);
            int sms=0; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
            int blocks=p->pages*4;
            if(blocks<=max_blocks*sms) {
                void* args[]={const_cast<rtp_llm::Params*>(p),cmark};
                cudaError_t ce=cudaLaunchCooperativeKernel((void*)rtp_llm::decode_cache_coop,dim3(p->pages,4),128,args,0,stream);
                if(ce!=cudaSuccess) return int(ce);
            } else {
                // grid too large for cooperative launch: fall back
                int64_t total=int64_t(p->batches)*4*((p->max_q+7)/8)*16;
                rtp_llm::mark_selected_pages<<<(unsigned)((total+255)/256),256,0,stream>>>(*p,cmark);
                rtp_llm::decode_working_cache<<<dim3(p->pages,4),128,0,stream>>>(*p,cmark);
            }
            err=cudaGetLastError();
            if(err!=cudaSuccess) return int(err);
        } else if(g_selfmark) {
            rtp_llm::decode_cache_selfmark<<<dim3(p->pages,4),128,0,stream>>>(*p);
            err=cudaGetLastError();
            if(err!=cudaSuccess) return int(err);
        } else {
        static uint8_t* mark=nullptr;
        static int mark_pages=0;
        if(!mark||mark_pages!=p->pages) {
            if(mark) cudaFree(mark);
            // [marks pages*4][epoch u32][stamps pages*4 u32]
            // [list i32 pages*4][counter u32]
            cudaMalloc(&mark,size_t(p->pages)*48+8);
            cudaMemsetAsync(mark,0,size_t(p->pages)*32+4,stream);
            cudaMemsetAsync(mark+size_t(p->pages)*32+4,0xff,size_t(p->pages)*16,stream);
            cudaMemsetAsync(mark+size_t(p->pages)*48+4,0,4,stream);
            mark_pages=p->pages;
        }
        int64_t total=int64_t(p->batches)*4*((p->max_q+7)/8)*16;
        // Compact-list cache grid: only for narrow grids whose selection
        // upper bound (batches*heads*tiles*16) is smaller than the whole
        // (page,head) pool - the full-page grid is then mostly empty CTAs
        // cycling at the hardware dispatch limit (~1ns/CTA). Wide grids and
        // full-coverage selections keep the legacy page grid bit-for-bit.
        // Compact-list cache grid: only where the full-page sweep dominates.
        // The legacy grid decodes pages in ascending order, which lets the
        // resident loader's stamps arrive early (near-zero spin); the list
        // grid's append order is scrambled, so it pays overlap latency plus
        // the CAS dedup (~2.9us). List wins only when the empty-CTA sweep
        // (pages*4 CTAs at ~1ns dispatch each) exceeds those costs -
        // empirically at pages*4 > 10000 (b8/kv128000: sweep 16us vs CAS
        // 2.9us; b4/kv65537: 8.2k CTAs is a wash).
        const int tiles=(p->max_q+7)/8;
        const bool wide=p->batches*tiles>g_wide_thresh_host;
        int64_t cap=int64_t(p->batches)*4*tiles*16;
        const bool use_list=g_listcache&&!wide&&cap<int64_t(p->pages)*4
                            &&int64_t(p->pages)*4>8000;
        if(use_list) {
            rtp_llm::mark_selected_pages_list<<<(unsigned)((total+255)/256),256,0,stream>>>(*p,mark,(int)cap);
        } else if(g_mark1) rtp_llm::mark_selected_pages4<<<(unsigned)((total+1023)/1024),256,0,stream>>>(*p,mark);
        else rtp_llm::mark_selected_pages<<<(unsigned)((total+255)/256),256,0,stream>>>(*p,mark);
        flags=reinterpret_cast<const uint32_t*>(mark+size_t(p->pages)*16);
        // Wide grids keep the serial (tail-trigger) path: null flags make
        // the loader skip the stamp spins entirely (r52: spins cost 1-2pt).
        if(wide) flags=nullptr;
        if(use_list)
            err=launch_with_pdl(rtp_llm::decode_working_cache_list,dim3((unsigned)cap),dim3(128),0,stream,*p,mark);
        else
            err=launch_with_pdl(rtp_llm::decode_working_cache,dim3(p->pages,4),dim3(128),0,stream,*p,mark);
        if(err!=cudaSuccess) return int(err);
        }
    }
#if MSA_RESIDENT
    rtp_llm::grouped_cutlass_resident<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::ResidentShared),stream>>>(*p);
#elif MSA_RESIDENT2
    rtp_llm::grouped_cutlass_resident2<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident2Shared),stream>>>(*p);
#elif MSA_RESIDENT3
    rtp_llm::grouped_cutlass_resident3<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident3Shared),stream>>>(*p);
#elif MSA_RESIDENT4
    rtp_llm::grouped_cutlass_resident4<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident4Shared),stream>>>(*p);
#elif MSA_RESIDENT5
    rtp_llm::grouped_cutlass_resident5<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident5Shared),stream>>>(*p);
#elif MSA_RESIDENT6
    rtp_llm::grouped_cutlass_resident6<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident6Shared),stream>>>(*p);
#elif MSA_RESIDENT7
    rtp_llm::grouped_cutlass_resident7<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident7Shared),stream>>>(*p);
#elif MSA_RESIDENT8
    rtp_llm::grouped_cutlass_resident8<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident8Shared),stream>>>(*p);
#elif MSA_RESIDENT11
    rtp_llm::grouped_cutlass_resident11<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident11Shared),stream>>>(*p);
#elif MSA_RESIDENT12
    if(p->k8) {
        err=launch_with_pdl(rtp_llm::grouped_cutlass_resident12,dim3((p->max_q+7)/8,p->batches*4),dim3(256),sizeof(rtp_llm::Resident12Shared),stream,*p,flags);
        if(err!=cudaSuccess) return int(err);
    } else {
        rtp_llm::grouped_cutlass_resident12<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident12Shared),stream>>>(*p,nullptr);
    }
#elif MSA_RESIDENT13
    rtp_llm::grouped_cutlass_resident13<<<dim3((p->max_q+7)/8,p->batches*4),256,sizeof(rtp_llm::Resident13Shared),stream>>>(*p);
#elif MSA_SPLIT_MERGE
    rtp_llm::grouped_cutlass_split_merge<<<dim3((p->max_q+7)/8,p->batches*4),512,sizeof(rtp_llm::SplitMergeShared),stream>>>(*p);
#elif MSA_DUAL_QUERY
    rtp_llm::grouped_cutlass_dual_query<<<dim3((p->max_q+15)/16,p->batches*4),512,sizeof(rtp_llm::DualQueryShared),stream>>>(*p);
#elif MSA_COMPACT
    rtp_llm::grouped_cutlass_compact<<<dim3((p->max_q+1)/2,p->batches*4),256,sizeof(rtp_llm::CompactShared),stream>>>(*p);
#elif MSA_LATE_MERGE
    rtp_llm::grouped_cutlass_late<<<dim3((p->max_q+3)/4,p->batches*4),512,sizeof(rtp_llm::LateShared),stream>>>(*p);
#else
    rtp_llm::grouped_cutlass_pipeline<<<dim3((p->max_q+7)/8,p->batches*4),512,sizeof(rtp_llm::PipelineShared),stream>>>(*p);
#endif
    return int(cudaGetLastError());
}

extern "C" int msa_resources(int* regs,int* shared,int* local) {
    cudaFuncAttributes attr;
#if MSA_RESIDENT
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::ResidentShared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT2
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident2);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident2Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT3
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident3);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident3Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT4
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident4);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident4Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT5
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident5);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident5Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT6
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident6);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident6Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT7
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident7);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident7Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT8
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident8);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident8Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT11
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident11);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident11Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT12
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident12);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident12Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT13
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident13);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident13Shared); *local=attr.localSizeBytes;
#elif MSA_RESIDENT9
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident9);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::Resident9Shared); *local=attr.localSizeBytes;
#elif MSA_SPLIT_MERGE
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_split_merge);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::SplitMergeShared); *local=attr.localSizeBytes;
#elif MSA_DUAL_QUERY
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_dual_query);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::DualQueryShared); *local=attr.localSizeBytes;
#elif MSA_COMPACT
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_compact);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::CompactShared); *local=attr.localSizeBytes;
#elif MSA_LATE_MERGE
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_late);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::LateShared); *local=attr.localSizeBytes;
#else
    auto err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_pipeline);
    *regs=attr.numRegs; *shared=sizeof(rtp_llm::PipelineShared); *local=attr.localSizeBytes;
#endif
    return int(err);
}

extern "C" int msa_decode_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_decode,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DecodeShared));
    if(err!=cudaSuccess) return 100+int(err);
    rtp_llm::grouped_cutlass_decode<<<dim3(p->max_q,4),256,sizeof(rtp_llm::DecodeShared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_decode_partial_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_decode_partial,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DecodePartialShared));
    if(err!=cudaSuccess) return 100+int(err);
    rtp_llm::grouped_cutlass_decode_partial<<<dim3(p->max_q,4,16),256,sizeof(rtp_llm::DecodePartialShared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_decode_m64_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_decode_m64,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DecodeM64Shared));
    if(err!=cudaSuccess) return 100+int(err);
    rtp_llm::grouped_cutlass_decode_m64<<<dim3(p->max_q,4,4),256,sizeof(rtp_llm::DecodeM64Shared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_m64_layout_probe(const uint8_t* q, const uint8_t* k, float* out, cudaStream_t stream) {
    rtp_llm::m64_layout_probe<<<1,256,0,stream>>>(q,k,out);
    return int(cudaGetLastError());
}

extern "C" int msa_warp_store_probe(float* out, cudaStream_t stream) {
    rtp_llm::warp_store_probe<<<1,256,0,stream>>>(out);
    return int(cudaGetLastError());
}

extern "C" int msa_warp_store_probe2(float* out, cudaStream_t stream) {
    rtp_llm::warp_store_probe2<<<1,256,0,stream>>>(out);
    return int(cudaGetLastError());
}

extern "C" int msa_n64_split_qk_probe(const uint8_t* q, const uint8_t* k, float* out, cudaStream_t stream) {
    rtp_llm::n64_split_qk_probe<<<1,128,0,stream>>>(q,k,out);
    return int(cudaGetLastError());
}

extern "C" int msa_m64_loadmap_probe(const uint8_t* q, const uint8_t* k, float* out, cudaStream_t stream) {
    rtp_llm::m64_loadmap_probe<<<1,256,0,stream>>>(q,k,out);
    return int(cudaGetLastError());
}

extern "C" int msa_pv_split_probe(const uint8_t* q, const uint8_t* v, float* out, cudaStream_t stream) {
    rtp_llm::pv_split_probe<<<1,128,0,stream>>>(q,v,out);
    return int(cudaGetLastError());
}

extern "C" int msa_mma_layout_probe(const uint8_t* a, const uint8_t* b, const float* c, float* d, cudaStream_t stream) {
    rtp_llm::mma_layout_probe<<<1,32,0,stream>>>(a,b,c,d);
    return int(cudaGetLastError());
}

extern "C" int msa_mma_kchain_probe(const uint8_t* a, const uint8_t* b, float* d, cudaStream_t stream) {
    rtp_llm::mma_kchain_probe<<<1,32,0,stream>>>(a,b,d);
    return int(cudaGetLastError());
}

extern "C" int msa_decode_warp_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::decode_warp_mma,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DecodeWarpShared));
    if(err!=cudaSuccess) return 100+int(err);
    rtp_llm::decode_warp_mma<<<dim3(p->max_q,4,1),128,sizeof(rtp_llm::DecodeWarpShared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_decode_pair_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::decode_pair_mma,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DecodePairShared));
    if(err!=cudaSuccess) return 100+int(err);
    rtp_llm::decode_pair_mma<<<dim3(p->max_q,4,1),256,sizeof(rtp_llm::DecodePairShared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_decode_pair_partial_launch(const rtp_llm::Params* p, int groups, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::decode_pair_partial,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::DecodePairShared));
    if(err!=cudaSuccess) return 100+int(err);
    rtp_llm::decode_pair_partial<<<dim3(p->max_q,4,groups),256,sizeof(rtp_llm::DecodePairShared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_resident12p_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident12p,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident12Shared));
    if(err!=cudaSuccess) return 100+int(err);
    err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident12p,cudaFuncAttributePreferredSharedMemoryCarveout,100);
    if(err!=cudaSuccess) return 100+int(err);
    if(p->k8) {
        static uint8_t* mark=nullptr;
        static int mark_pages=0;
        if(!mark||mark_pages<p->pages) {
            if(mark) cudaFree(mark);
            cudaMalloc(&mark,size_t(p->pages)*32+4);
            mark_pages=p->pages;
        }
        cudaMemsetAsync(mark,0,size_t(p->pages)*32+4,stream);
        int64_t total=int64_t(p->batches)*4*((p->max_q+7)/8)*16;
        rtp_llm::mark_selected_pages4<<<(unsigned)((total+1023)/1024),256,0,stream>>>(*p,mark);
        rtp_llm::decode_working_cache<<<dim3(p->pages,4),128,0,stream>>>(*p,mark);
        err=cudaGetLastError();
        if(err!=cudaSuccess) return int(err);
    }
    rtp_llm::grouped_cutlass_resident12p<<<dim3(148,2),256,sizeof(rtp_llm::Resident12Shared),stream>>>(*p);
    return int(cudaGetLastError());
}

extern "C" int msa_resident12g2_launch(const rtp_llm::Params* p, cudaStream_t stream) {
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident12g2,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::Resident12Shared));
    if(err!=cudaSuccess) return 100+int(err);
    err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident12g2,cudaFuncAttributePreferredSharedMemoryCarveout,100);
    if(err!=cudaSuccess) return 100+int(err);
    const uint32_t* flags=nullptr;
    if(p->k8) {
        static uint8_t* mark=nullptr;
        static int mark_pages=0;
        if(!mark||mark_pages<p->pages) {
            if(mark) cudaFree(mark);
            cudaMalloc(&mark,size_t(p->pages)*32+4);
            cudaMemsetAsync(mark,0,size_t(p->pages)*32+4,stream);
            mark_pages=p->pages;
        }
        int64_t total=int64_t(p->batches)*4*((p->max_q+7)/8)*16;
        if(g_mark1) rtp_llm::mark_selected_pages4<<<(unsigned)((total+1023)/1024),256,0,stream>>>(*p,mark);
        else rtp_llm::mark_selected_pages<<<(unsigned)((total+255)/256),256,0,stream>>>(*p,mark);
        flags=reinterpret_cast<const uint32_t*>(mark+size_t(p->pages)*16);
        err=launch_with_pdl(rtp_llm::decode_working_cache,dim3(p->pages,4),dim3(128),0,stream,*p,mark);
        if(err!=cudaSuccess) return int(err);
    }
    if(p->k8) {
        err=launch_with_pdl(rtp_llm::grouped_cutlass_resident12g2,dim3((p->max_q+7)/8,p->batches*4,g_g2_groups),dim3(256),sizeof(rtp_llm::Resident12Shared),stream,*p,flags);
        if(err!=cudaSuccess) return int(err);
    } else {
        rtp_llm::grouped_cutlass_resident12g2<<<dim3((p->max_q+7)/8,p->batches*4,g_g2_groups),256,sizeof(rtp_llm::Resident12Shared),stream>>>(*p,nullptr);
    }
    return int(cudaGetLastError());
}

extern "C" int msa_resident_cta_count() {
    int count=0;
    auto err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(rtp_llm::ResidentShared));
    if(err!=cudaSuccess) return -int(err);
    err=cudaFuncSetAttribute(rtp_llm::grouped_cutlass_resident,cudaFuncAttributePreferredSharedMemoryCarveout,100);
    if(err!=cudaSuccess) return -int(err);
    err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count,rtp_llm::grouped_cutlass_resident,256,sizeof(rtp_llm::ResidentShared));
    return err==cudaSuccess?count:-int(err);
}

extern "C" int msa_resident_resource_details(int* values) {
    cudaFuncAttributes attr;
    cudaDeviceProp prop;
    int device=0;
    auto err=cudaGetDevice(&device);
    if(err!=cudaSuccess) return int(err);
    err=cudaGetDeviceProperties(&prop,device);
    if(err!=cudaSuccess) return int(err);
    err=cudaFuncGetAttributes(&attr,rtp_llm::grouped_cutlass_resident);
    if(err!=cudaSuccess) return int(err);
    values[0]=attr.numRegs;values[1]=attr.sharedSizeBytes;values[2]=attr.maxDynamicSharedSizeBytes;
    values[3]=attr.preferredShmemCarveout;values[4]=prop.regsPerMultiprocessor;values[5]=prop.sharedMemPerMultiprocessor;
    values[6]=prop.maxThreadsPerMultiProcessor;values[7]=attr.maxThreadsPerBlock;
    err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(values+8,rtp_llm::grouped_cutlass_resident,256,0);
    if(err!=cudaSuccess) return int(err);
    err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(values+9,rtp_llm::grouped_cutlass_resident,128,0);
    return int(err);
}
