// m3idx_types_m31.cuh -- M3.1 index-geometry constants.
//
// Ported from the read-only M3.0 reference
//   /home/liukan.lk/m3pro-128blk-20261009/include/m3idx_types.cuh
// (namespace m3idx, kIdxQHeads=16, kPageSize=16) with the M3.1 geometry
// changes frozen in the port plan:
//   * kIdxQHeads 16 -> 4   (H=4 index heads; CP shards heads so the LOCAL
//                            head count is what the kernels see)
//   * kPageSize   16 -> 128 (one page == one 128-token contiguous block;
//                            page-table indirection stays via kv_indices /
//                            block_table, one logical block == one physical
//                            page)
//   * kIdxKVHeads = 1, kIdxDim = 128 unchanged
//   * all chunk metadata pointers are int32 (M3.0 read cu_* as int64_t*)
//
// OurPrefillParams / OurDecodeParams are NOT redefined here: the single
// source of truth is the frozen C ABI header our_score_abi.h, which this
// file includes so kernel-side users get both from one include.
#pragma once

#include <cstdint>
#include <cstddef>

#include "our_score_abi.h"

namespace m3idx31 {

constexpr int kIdxQHeads = 4;    // IQ heads (M3.1: 4 x 128; was 16 in M3.0)
constexpr int kIdxKVHeads = 1;   // IK shared head (GQA-style sharing, unchanged)
constexpr int kIdxDim = 128;     // indexer head dim (unchanged)
constexpr int kPageSize = 128;   // tokens per page == one scored block (was 16)

} // namespace m3idx31
