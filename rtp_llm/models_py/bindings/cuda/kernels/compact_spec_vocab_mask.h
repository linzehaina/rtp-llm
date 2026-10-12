#pragma once

#include <torch/torch.h>

namespace rtp_llm {

// Packed int32 [rows, ceil(vocab / 32)], low bit first, 1 means allow.
// Uses the legacy masked_fill_ -FLT_MAX sentinel and checked dtype conversion.
// Applies directly to logits in place without materializing a dense bool mask.
void applyCompactSpecVocabMask(const torch::Tensor& logits, const torch::Tensor& bitmask);

// Draft [B,P] contiguous int32/int64; boundaries [B,3] int32 (active, begin, end).
// Writes fresh packed mask and cap; preserves W*32 bounds and post-cap allow rows.
void buildStaticThinkSpecMask(const torch::Tensor& draft, const torch::Tensor& boundaries,
                             const torch::Tensor& bitmask, const torch::Tensor& cap);

}  // namespace rtp_llm
