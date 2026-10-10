#pragma once

namespace rtp_llm {
struct alignas(128) CompactShared {
    alignas(128) uint8_t q[8192],k[16384],v[16384],p[8192];
    alignas(128) uint32_t partial[7][64][32];
    float lp[16][32],inv[32];
    alignas(8) uint64_t kv_ready,kv_free,p_ready,score_ready,o_ready,o_free,stats_free,done;
    uint64_t spill_ready,spill_free,gather_ready,gather_free;
    uint32_t tmem;
};
static_assert(sizeof(CompactShared)*2<=233472,"compact late merge must fit two CTAs per SM");

template<int Id>
__device__ __forceinline__ void pair_warp_sync() {
    asm volatile("bar.sync %0,64;"::"n"(Id):"memory");
}
template<size_t... I>
__device__ __forceinline__ void compact_store_impl(uint32_t addr,const uint32_t* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_STORE_16dp32b32x::copy(v[I]...,addr);
}
template<size_t... I>
__device__ __forceinline__ void compact_load_impl(uint32_t addr,uint32_t* v,std::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_16dp32b32x::copy(addr,v[I]...);
}
__device__ __forceinline__ uint32_t compact_partial_address(uint32_t t,int page,int warp) {
    // M64 MMA writes only the lower 16 datapaths of each 32-datapath group.
    int group=page%4;
    return t+(page/4)*128+(group%2)*64+((16+(group/2)*64+warp*32)<<16);
}
__device__ __forceinline__ void compact_load16(uint32_t addr,uint32_t* v) {
    asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16],32;"
        :"=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]),"=r"(v[4]),"=r"(v[5]),"=r"(v[6]),"=r"(v[7]),
         "=r"(v[8]),"=r"(v[9]),"=r"(v[10]),"=r"(v[11]),"=r"(v[12]),"=r"(v[13]),"=r"(v[14]),"=r"(v[15]):"r"(addr));
}

__global__ __launch_bounds__(256,2) void grouped_cutlass_compact(Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    auto& s=*reinterpret_cast<CompactShared*>(smem);
    int tid=threadIdx.x,local=tid%64,warp=local/32,lane=local%32,row=warp*16+lane%16,half=lane/16;
    int batch=blockIdx.y/4,head=blockIdx.y%4,start=p.cuq[batch],stop=p.cuq[batch+1],qbase=start+int(blockIdx.x)*2;
    if(qbase>=stop) return;
    int length=p.used?p.used[batch]:p.cuk[batch+1]-p.cuk[batch];
    if(tid==0) {
        cutlass::arch::ClusterBarrier::init(&s.kv_ready,1);cutlass::arch::ClusterBarrier::init(&s.kv_free,1);
        cutlass::arch::ClusterBarrier::init(&s.p_ready,1);cutlass::arch::ClusterBarrier::init(&s.score_ready,1);
        cutlass::arch::ClusterBarrier::init(&s.o_ready,1);cutlass::arch::ClusterBarrier::init(&s.o_free,1);
        cutlass::arch::ClusterBarrier::init(&s.stats_free,1);cutlass::arch::ClusterBarrier::init(&s.done,1);
        cutlass::arch::ClusterBarrier::init(&s.spill_ready,1);cutlass::arch::ClusterBarrier::init(&s.spill_free,1);
        cutlass::arch::ClusterBarrier::init(&s.gather_ready,1);cutlass::arch::ClusterBarrier::init(&s.gather_free,1);
    }
    if(tid<32) {
        cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    for(int i=tid;i<2048;i+=256) {
        int r=i/32,d=i%32*4,q=qbase+r/16;
        uint32_t value=(r<32&&q<stop)?*reinterpret_cast<const uint32_t*>(p.q+int64_t(q)*p.qn+(head*16+r%16)*p.qh+d):0;
        *reinterpret_cast<uint32_t*>(s.q+Layout64{}(cute::make_coord(r,d)))=value;
        *reinterpret_cast<uint32_t*>(s.p+i*4)=0;
    }
    cutlass::arch::fence_barrier_init();cutlass::arch::fence_view_async_shared();__syncthreads();
    uint32_t t=s.tmem,addr=uint32_t(warp*32)<<16;
    if(tid<64||(tid>=128&&tid<192)) {
        if(tid<64) {
            int query=qbase+warp,limit=query-start+length-(stop-start);
            for(int j=0;j<16;++j) {
                cutlass::arch::ClusterBarrier::wait(&s.score_ready,j%2);after_sync();
                float v[64];load64_m64(t+addr,v);cutlass::arch::fence_view_async_tmem_load();
                int block=p.topk[head*p.th+int64_t(qbase)*p.tq+j];
                bool valid=block>=0&&block<p.max_pages&&block*128<length;
                int page=valid?p.table[batch*p.tb+block]:-1;
                valid=valid&&page>=0&&page<p.pages&&query<stop;
                int last=valid?min(length-1,limit)-block*128:-1;
                float maxima[8];
                #pragma unroll
                for(int c=0;c<8;++c) maxima[c]=-CUDART_INF_F;
                #pragma unroll
                for(int c=0;c<64;++c) {
                    float masked;
                    asm("{ .reg .pred vis; setp.le.s32 vis,%3,%4; selp.f32 %0,%1,%2,vis; }"
                        :"=f"(masked):"f"(v[c]),"f"(-CUDART_INF_F),"r"(half*64+c),"r"(last));
                    v[c]=masked;maxima[c%8]=max_native(maxima[c%8],masked);
                }
                float maximum=max_native(max_native(max_native(maxima[0],maxima[1]),max_native(maxima[2],maxima[3])),
                                         max_native(max_native(maxima[4],maxima[5]),max_native(maxima[6],maxima[7])));
                maximum=max_native(maximum,__shfl_xor_sync(0xffffffff,maximum,16));
                float shift=(maximum==-CUDART_INF_F?0.f:maximum)*(-p.alpha);
                #pragma unroll
                for(int c=0;c<64;c+=2) {
                    affine_pair(v[c],v[c+1],p.alpha,shift);
                    float a=exp_native(v[c]),b=exp_native(v[c+1]);
                    if(c%16>=12) {
                        float ea=exp_emulated(v[c]),eb=exp_emulated(v[c+1]);int em=c<32?half:1-half;
                        asm("{ .reg .pred sel; setp.ne.u32 sel,%4,0; selp.f32 %0,%2,%0,sel; selp.f32 %1,%3,%1,sel; }"
                            :"+f"(a),"+f"(b):"f"(ea),"f"(eb),"r"(em));
                    }
                    v[c]=a;v[c+1]=b;
                }
                float sums[8]={};
                #pragma unroll
                for(int i=0;i<8;++i) {
                    #pragma unroll
                    for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[i*8+c],v[i*8+c+1]);
                }
                #pragma unroll
                for(int c=0;c<8;++c) sums[c]=__shfl_sync(0xffffffff,sums[c],lane%16);
                #pragma unroll
                for(int i=0;i<8;++i) {
                    #pragma unroll
                    for(int c=0;c<8;c+=2) add_pair(sums[c],sums[c+1],v[i*8+c],v[i*8+c+1]);
                }
                float den=((sums[0]+sums[2])+(sums[4]+sums[6]))+((sums[1]+sums[3])+(sums[5]+sums[7]));
                den=__shfl_sync(0xffffffff,den,lane%16+16);
                if(MSA_TRACE&&p.debug&&blockIdx.x==0&&blockIdx.y==0&&local==0&&j==0) printf("softmax block=%d valid=%d max=%f den=%f prob=%f\n",block,int(valid),maximum,den,v[0]);
                if(j>0) cutlass::arch::ClusterBarrier::wait(&s.stats_free,(j-1)%2);
                #pragma unroll
                for(int c=0;c<64;c+=4) *reinterpret_cast<uint32_t*>(s.p+Layout64{}(cute::make_coord(row,half*64+c)))=pack_probability(v[c],v[c+1],v[c+2],v[c+3]);
                if(half==0) {
                    s.inv[row]=rcp_native(den>0.f?den:1.f);
                    s.lp[j][row]=maximum==-CUDART_INF_F?-CUDART_INF_F:fmaf(maximum,p.alpha,log_native(den));
                }
                cutlass::arch::fence_view_async_shared();pair_warp_sync<1>();
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.p_ready);
            }
        } else {
            for(int j=0;j<15;++j) {
                cutlass::arch::ClusterBarrier::wait(&s.o_ready,j%2);after_sync();
                float values[64];load64_m64(t+128+addr,values);cutlass::arch::fence_view_async_tmem_load();
                float inv=s.inv[row];pair_warp_sync<2>();
                if(MSA_TRACE&&p.debug&&blockIdx.x==0&&blockIdx.y==0&&local==0&&j==0) printf("merge pv=%f inv=%f\n",values[0],inv);
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.stats_free);
                uint32_t packed[32];
                #pragma unroll
                for(int c=0;c<32;++c) packed[c]=normalize_bf16_pair(values[c*2],values[c*2+1],inv);
                if(MSA_TRACE&&p.debug&&blockIdx.x==0&&blockIdx.y==0) {
                    auto* d=reinterpret_cast<float*>(p.debug);
                    #pragma unroll
                    for(int c=0;c<32;++c) {
                        d[(j*32+row)*128+half*64+c*2]=__uint_as_float(packed[c]<<16);
                        d[(j*32+row)*128+half*64+c*2+1]=__uint_as_float(packed[c]&0xffff0000);
                    }
                }
                if(j<8&&j%4<2) {
                    compact_store_impl(compact_partial_address(t,j,warp),packed,std::make_index_sequence<32>{});
                    cutlass::arch::fence_view_async_tmem_store();
                } else if(j<8) {
                    #pragma unroll
                    for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(&s.partial[0][local][c^((local&7)*4)])=*reinterpret_cast<uint4*>(packed+c);
                    pair_warp_sync<2>();
                    if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.spill_ready);
                    cutlass::arch::ClusterBarrier::wait(&s.spill_free,(j%4-2)%2);
                } else {
                    #pragma unroll
                    for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(&s.partial[j-8][local][c^((local&7)*4)])=*reinterpret_cast<uint4*>(packed+c);
                }
                before_sync();pair_warp_sync<2>();
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.o_free);
            }
            cutlass::arch::ClusterBarrier::wait(&s.o_ready,1);after_sync();
            float last_values[64];load64_m64(t+128+addr,last_values);cutlass::arch::fence_view_async_tmem_load();
            float last_inv=s.inv[row];
            uint32_t last_page[32];
            #pragma unroll
            for(int c=0;c<32;++c) last_page[c]=normalize_bf16_pair(last_values[c*2],last_values[c*2+1],last_inv);
            float weights[16],maximum=-CUDART_INF_F,total=0.f;
            #pragma unroll
            for(int j=0;j<16;++j) {weights[j]=s.lp[j][row];maximum=max_native(maximum,weights[j]);}
            float safe=maximum==-CUDART_INF_F?0.f:maximum;
            #pragma unroll
            for(int j=0;j<16;++j) {weights[j]=exp_native(weights[j]-safe);total+=weights[j];}
            float inv=rcp_native(total>0.f?total:1.f);
            auto* output=reinterpret_cast<__nv_bfloat16*>(s.k);
            #pragma unroll
            for(int part=0;part<2;++part) {
            float accum[32]={};
            for(int j=0;j<15;++j) {
                uint32_t packed[16];
                if(j<8&&j%4<2) {
                    compact_load16(compact_partial_address(t,j,warp)+part*16,packed);
                    cutlass::arch::fence_view_async_tmem_load();
                } else if(j<8) {
                    cutlass::arch::ClusterBarrier::wait(&s.gather_ready,(j%4-2)%2);
                    auto* stage=reinterpret_cast<uint32_t*>(s.q);
                    #pragma unroll
                    for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(packed+c)=*reinterpret_cast<uint4*>(stage+local*32+((part*16+c)^((local&7)*4)));
                    pair_warp_sync<2>();
                    if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.gather_free);
                } else {
                    #pragma unroll
                    for(int c=0;c<16;c+=4) *reinterpret_cast<uint4*>(packed+c)=*reinterpret_cast<uint4*>(&s.partial[j-8][local][(part*16+c)^((local&7)*4)]);
                }
                if(MSA_TRACE&&p.debug&&blockIdx.x==0&&blockIdx.y==0) {
                    auto* d=reinterpret_cast<float*>(p.debug)+65536;
                    #pragma unroll
                    for(int c=0;c<16;++c) {
                        d[(j*32+row)*128+half*64+part*32+c*2]=__uint_as_float(packed[c]<<16);
                        d[(j*32+row)*128+half*64+part*32+c*2+1]=__uint_as_float(packed[c]&0xffff0000);
                    }
                }
                #pragma unroll
                for(int c=0;c<16;++c) accumulate_bf16_pair(accum[c*2],accum[c*2+1],packed[c],weights[j]);
            }
            #pragma unroll
            for(int c=0;c<16;++c) accumulate_bf16_pair(accum[c*2],accum[c*2+1],last_page[part*16+c],weights[15]);
            #pragma unroll
            for(int c=0;c<32;c+=2) *reinterpret_cast<uint32_t*>(output+OutputLayout64{}(cute::make_coord(row,half*64+part*32+c)))=normalize_bf16_pair(accum[c],accum[c+1],inv);
            }
            int query=qbase+warp;
            if(half==0&&query<stop) p.lse[int64_t(query)*64+head*16+lane%16]=total>0.f?(maximum+log_native(total))*0.6931471805599453f:-CUDART_INF_F;
            pair_warp_sync<2>();if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.done);
        }
    } else {
        if(tid>=64&&tid<128) {
            using Copy=cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
            for(int j=0;j<16;++j) {
                if(j>0) cutlass::arch::ClusterBarrier::wait(&s.kv_free,(j-1)%2);
                int block=p.topk[head*p.th+int64_t(qbase)*p.tq+j];
                bool valid=block>=0&&block<p.max_pages&&block*128<length;
                int page=valid?p.table[batch*p.tb+block]:-1;
                valid=valid&&page>=0&&page<p.pages;
                int safe_page=valid?page:0;
                for(int i=local;i<1024;i+=64) {
                    int token=i/8,d=i%8*16;bool present=valid&&block*128+token<length;
                    int64_t offset=(int64_t(safe_page)*4+head)*16384+token*128+d;
                    Copy::copy(*reinterpret_cast<const uint4*>(p.k8+offset),*reinterpret_cast<uint4*>(s.k+Layout{}(cute::make_coord(token,d))),present);
                    Copy::copy(*reinterpret_cast<const uint4*>(p.v8+offset),*reinterpret_cast<uint4*>(s.v+VLayout{}(cute::make_coord(d,token))),present);
                }
                cute::cp_async_fence();cute::cp_async_wait<0>();cutlass::arch::fence_view_async_shared();pair_warp_sync<3>();
                if(MSA_TRACE&&p.debug&&blockIdx.x==0&&blockIdx.y==0&&local==0&&j==0) printf("loader page=%d valid=%d k=%x v=%x\n",page,int(valid),*reinterpret_cast<uint32_t*>(s.k),*reinterpret_cast<uint32_t*>(s.v));
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.kv_ready);
                if(j<8&&j%4>=2) {
                    cutlass::arch::ClusterBarrier::wait(&s.spill_ready,(j%4-2)%2);
                    uint32_t packed[32];
                    #pragma unroll
                    for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(packed+c)=*reinterpret_cast<uint4*>(&s.partial[0][local][c^((local&7)*4)]);
                    compact_store_impl(compact_partial_address(t,j,warp),packed,std::make_index_sequence<32>{});
                    cutlass::arch::fence_view_async_tmem_store();before_sync();pair_warp_sync<3>();
                    if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.spill_free);
                }
            }
            cutlass::arch::ClusterBarrier::wait(&s.kv_free,1);
            for(int step=0;step<8;++step) {
                if(step>0) cutlass::arch::ClusterBarrier::wait(&s.gather_free,(step-1)%2);
                int j=(step%4<2)?step%4+2:step%4+4;
                uint32_t packed[32];
                compact_load_impl(compact_partial_address(t,j,warp),packed,std::make_index_sequence<32>{});
                cutlass::arch::fence_view_async_tmem_load();
                auto* stage=reinterpret_cast<uint32_t*>(s.q);
                #pragma unroll
                for(int c=0;c<32;c+=4) *reinterpret_cast<uint4*>(stage+local*32+(c^((local&7)*4)))=*reinterpret_cast<uint4*>(packed+c);
                pair_warp_sync<3>();
                if(local==0) cutlass::arch::ClusterBarrier::arrive(&s.gather_ready);
            }
        } else if(tid<224) {
            uint64_t dq=descriptor64(s.q),dk=descriptor(s.k),dp=descriptor64(s.p);
            uint64_t dv=cute::UMMA::make_umma_desc<cute::UMMA::Major::MN>(cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<F8*>(s.v)),VLayout{}));
            uint64_t inst=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::K>();
            uint64_t inst_pv=cute::UMMA::make_runtime_instr_desc<F8,F8,float,64,128,cute::UMMA::Major::K,cute::UMMA::Major::MN>();
            for(int j=0;j<16;++j) {
                cutlass::arch::ClusterBarrier::wait(&s.kv_ready,j%2);after_sync();
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dq+k/16,dk+k/16,t,k!=0,inst);
                cutlass::arch::umma_arrive(&s.score_ready);
                cutlass::arch::ClusterBarrier::wait(&s.p_ready,j%2);
                if(j>0) cutlass::arch::ClusterBarrier::wait(&s.o_free,(j-1)%2);
                after_sync();
                #pragma unroll
                for(int k=0;k<128;k+=32) QKMma::fma(dp+k/16,dv+VLayout{}(cute::make_coord(0,k))/16,t+128,k!=0,inst_pv);
                cutlass::arch::umma_arrive(&s.o_ready);cutlass::arch::umma_arrive(&s.kv_free);
            }
            cutlass::arch::ClusterBarrier::wait(&s.done,0);
        }
    }
    __syncthreads();
    auto* output=reinterpret_cast<__nv_bfloat16*>(s.k);
    for(int i=tid;i<512;i+=256) {
        int r=i/16,d=i%16*8,q=qbase+r/16;
        if(q<stop) *reinterpret_cast<uint4*>(p.out+(int64_t(q)*64+head*16+r%16)*128+d)=*reinterpret_cast<uint4*>(output+OutputLayout64{}(cute::make_coord(r,d)));
    }
    before_sync();__syncthreads();if(tid<32) cute::TMEM::Allocator1Sm().free(t,256);
}
} // namespace rtp_llm
