#pragma once

// Probe for mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 fragment
// layouts. Verifies the hypothesized per-lane (row, col) mappings for A, B and
// C/D against a reference matrix product supplied from the host.

namespace rtp_llm {

// Hypothesis H1:
//   A[16,32]: lane l, reg r (0..3), byte j (0..3):
//     row = l/4 + 8*(r%2)          (a0,a2 -> l/4 ; a1,a3 -> l/4+8)
//     k   = (l%4)*4 + j + 16*(r/2) (a0,a1 -> k<16 ; a2,a3 -> k>=16)
//   B[32,8]: lane l, reg r (0..1), byte j:
//     n   = l/4
//     k   = (l%4)*4 + j + 16*r
//   C/D[16,8]: lane l, element i (0..3):
//     row = l/4 + 8*(i/2)
//     col = (l%4)*2 + (i%2)
__global__ __launch_bounds__(32,1) void mma_layout_probe(const uint8_t* a,
                                                          const uint8_t* b,
                                                          const float* c,
                                                          float* d) {
    const int lane=threadIdx.x;
    uint32_t ar[4],br[2];
    #pragma unroll
    for(int r=0;r<4;++r) {
        uint8_t bytes[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            int row=lane/4+8*(r%2);
            int k=(lane%4)*4+j+16*(r/2);
            bytes[j]=a[row*32+k];
        }
        ar[r]=*reinterpret_cast<uint32_t*>(bytes);
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        uint8_t bytes[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            int n=lane/4;
            int k=(lane%4)*4+j+16*r;
            bytes[j]=b[k*8+n];
        }
        br[r]=*reinterpret_cast<uint32_t*>(bytes);
    }
    float cr[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
        int row=lane/4+8*(i/2);
        int col=(lane%4)*2+(i%2);
        cr[i]=c[row*8+col];
    }
    float dr[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(dr[0]),"=f"(dr[1]),"=f"(dr[2]),"=f"(dr[3])
        : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),
          "r"(br[0]),"r"(br[1]),
          "f"(cr[0]),"f"(cr[1]),"f"(cr[2]),"f"(cr[3]));
    #pragma unroll
    for(int i=0;i<4;++i) {
        int row=lane/4+8*(i/2);
        int col=(lane%4)*2+(i%2);
        d[row*8+col]=dr[i];
    }
}

// Second probe: chains four k-steps (k=0..127) accumulating into the same D
// fragment, mirroring the QK^T (K=128) shape used by the decode kernels.
__global__ __launch_bounds__(32,1) void mma_kchain_probe(const uint8_t* a,
                                                         const uint8_t* b,
                                                         float* d) {
    const int lane=threadIdx.x;
    uint32_t ar[16];
    #pragma unroll
    for(int s=0;s<4;++s) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            uint8_t bytes[4];
            #pragma unroll
            for(int j=0;j<4;++j) {
                int row=lane/4+8*(r%2);
                int k=s*32+(lane%4)*4+j+16*(r/2);
                bytes[j]=a[row*128+k];
            }
            ar[s*4+r]=*reinterpret_cast<uint32_t*>(bytes);
        }
    }
    float acc[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int s=0;s<4;++s) {
        uint32_t br[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            uint8_t bytes[4];
            #pragma unroll
            for(int j=0;j<4;++j) {
                int n=lane/4;
                int k=s*32+(lane%4)*4+j+16*r;
                bytes[j]=b[k*8+n];
            }
            br[r]=*reinterpret_cast<uint32_t*>(bytes);
        }
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(ar[s*4+0]),"r"(ar[s*4+1]),"r"(ar[s*4+2]),"r"(ar[s*4+3]),
              "r"(br[0]),"r"(br[1]));
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
        int row=lane/4+8*(i/2);
        int col=(lane%4)*2+(i%2);
        d[row*8+col]=acc[i];
    }
}

} // namespace rtp_llm
