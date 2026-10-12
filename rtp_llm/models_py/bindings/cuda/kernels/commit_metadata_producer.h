#pragma once

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {

// Full table overwrite plus live attention metadata production in one launch.
// Caller supplies contiguous int32 vectors of B source / C destination rows,
// C+1 cumulative destinations, current-device pointers and a correctly ordered
// stream. All source owners must survive the asynchronous launch; all written
// destinations must be disjoint from every source and other written destination.
// Optional tail-only vectors may be omitted. No storage is allocated or reused.
struct CudaGraphCommitMetadataParams {
    CudaGraphTableOverwriteParams tables;
    const int32_t* input_lengths = nullptr;
    const int32_t* prefix_lengths = nullptr;
    const int32_t* sequence_lengths_plus_1 = nullptr; // null: derive prefix+1
    int32_t* output_input_lengths = nullptr;
    int32_t* output_prefix_lengths = nullptr;
    int32_t* output_cu_seqlens = nullptr;
    int32_t* output_cu_kv_seqlens = nullptr;
    int32_t* output_sequence_lengths_plus_1 = nullptr;
    int32_t* output_sequence_lengths = nullptr; // preserve live, zero padded
    int32_t* output_decode_cu_seqlens = nullptr; // preserve [0,B], tail last_valid
    int batch_size = 0;
    int batch_capacity = 0;
    int last_valid = 0; // only cu/decode-cu padded tails; not cu_kv tail
};

// cu[0..B] = input prefix sums; cu[B+1..C] = last_valid.
// cu_kv[0..B] = prefix sums of input+prefix; cu_kv[B+1..C] is untouched.
// int32 outputs retain modulo-2^32 arithmetic. B=0 is supported natively.
// Only host POD checks and one kernel launch; no allocation, CUDA query or sync.
void invokeCudaGraphCommitMetadataProducer(CudaGraphCommitMetadataParams params, cudaStream_t stream);

} // namespace rtp_llm
