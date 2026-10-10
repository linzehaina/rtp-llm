#pragma once

#include <torch/extension.h>

namespace torch_ext {

// FP32 scores [heads, packed_queries, blocks], INT32 output [heads, queries, K].
// Last dimensions are contiguous; head/query pitches may include retained capacity.
// cu_seqlens is monotonic, starts at zero and ends at queries. prefix_lens >= 0,
// prefix[b] + query_length[b] <= blocks * 128, max_query_length covers every segment.
// CUDA metadata is read at execution time, including during CUDA Graph replay.
// Selected IDs follow descending scores; equal scores select the smaller logical block ID.
// NaN becomes -1e30. Initial blocks become 1e30, then local blocks become 1e29.
// Missing slots are -1. Optional block tables put the current block last.
// Returns INT32 flags [heads, queries] for selected/cutoff ties, infinities, or
// a cutoff affected by legacy -1e30 padding. The caller reuses historical sorting.
torch::Tensor minimax_prefill_topk(const torch::Tensor& scores,
                                   torch::Tensor&       topk_idx,
                                   torch::Tensor&       block_tables,
                                   torch::Tensor&       output_seq_lens,
                                   const torch::Tensor& cu_seqlens,
                                   const torch::Tensor& prefix_lens,
                                   int64_t              max_query_length,
                                   int64_t              topk,
                                   int64_t              init_blocks,
                                   int64_t              local_blocks,
                                   bool                 emit_block_table,
                                   int64_t              num_pages);

}  // namespace torch_ext
