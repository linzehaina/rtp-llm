#pragma once

#include <array>
#include <vector>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/table_overwrite_plan.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/commit_metadata_producer.h"

namespace rtp_llm {
namespace detail {

inline bool isCompactCommitMetadataSource(const PyModelInputs& inputs) {
    const auto& src = inputs.attention_inputs;
    const auto nonempty = [](const torch::Tensor& tensor) { return tensor.defined() && tensor.numel() > 0; };
    // Compact producer sources must not silently drop fields the legacy copy
    // path would publish. Existing derived metadata uses complete legacy prep.
    if (nonempty(src.cu_seqlens) || nonempty(src.cu_kv_seqlens) || nonempty(src.decode_cu_seqlens_d)
        || nonempty(inputs.combo_position_ids) || nonempty(src.combo_position_ids)
        || nonempty(inputs.embedding_inputs.combo_tokens_type_ids)
        || nonempty(inputs.embedding_inputs.text_tokens_mask)
        || nonempty(inputs.multimodal_inputs.mm_features_locs)
        || nonempty(inputs.bert_embedding_inputs.combo_position_ids)
        || nonempty(inputs.bert_embedding_inputs.combo_tokens_type_ids)
        || nonempty(inputs.bert_embedding_inputs.position_encoding)
        || nonempty(inputs.bert_embedding_inputs.token_type_embedding)
        || !inputs.multimodal_inputs.multimodal_features.empty() || !inputs.multimodal_inputs.mm_extra_input.empty()) {
        return false;
    }
    return src.is_prefill && src.is_target_verify && !src.is_ragged_target_verify
           && (!src.sequence_lengths.defined() || src.sequence_lengths.numel() == 0);

}

// Host-only, all-or-nothing validation. No allocation on device, dynamic
// device reads, or writes to capture storage before the complete plan passes.
inline bool makeCommitMetadataProducerPlan(const PyModelInputs& inputs,
                                          const PyModelInputs& captured,
                                          const CudaGraphState& state,
                                          CudaGraphCommitMetadataParams& output) {
    output = {};
    const auto& src = inputs.attention_inputs;
    const auto& dst = captured.attention_inputs;
    if (!isCompactCommitMetadataSource(inputs)) return false;
    const int b = state.current_batch_size;
    if (b <= 0 || !src.is_prefill || !src.is_target_verify || !dst.input_lengths.defined()
        || dst.input_lengths.dim() != 1 || dst.input_lengths.size(0) > std::numeric_limits<int>::max()) {
        return false;
    }
    const int capacity = static_cast<int>(dst.input_lengths.size(0));
    const int device = dst.input_lengths.is_cuda() ? dst.input_lengths.get_device() : -1;
    const auto vector_ok = [device](const torch::Tensor& tensor, int64_t size) {
        return tensor.defined() && tensor.is_cuda() && tensor.get_device() == device
               && tensor.scalar_type() == torch::kInt32 && tensor.dim() == 1
               && tensor.is_contiguous() && tensor.numel() == size;
    };
    if (capacity < b || !vector_ok(src.input_lengths, b) || !vector_ok(src.prefix_lengths, b)
        || (src.sequence_lengths.defined() && src.sequence_lengths.numel() != 0)
        || !vector_ok(dst.input_lengths, capacity) || !vector_ok(dst.prefix_lengths, capacity)
        || !vector_ok(dst.cu_seqlens, capacity + int64_t{1})
        || !vector_ok(dst.cu_kv_seqlens, capacity + int64_t{1})
        || !vector_ok(dst.sequence_lengths_plus_1_d, capacity)
        || (src.sequence_lengths_plus_1_d.defined() && !vector_ok(src.sequence_lengths_plus_1_d, b))) {
        return false;
    }
    if (capacity > b && (!vector_ok(dst.sequence_lengths, capacity)
                         || !vector_ok(dst.decode_cu_seqlens_d, capacity + int64_t{1}))) {
        return false;
    }
    const size_t groups = dst.kv_cache_kernel_block_id_device_by_group.size();
    if ((groups && groups != src.kv_cache_kernel_block_id_device_by_group.size())
        || groups > kMaxCudaGraphPrepareFillRegions - 2) {
        return false;
    }
    std::array<GraphTableTensorPair, kMaxCudaGraphPrepareFillRegions> tables{};
    tables[0] = {&src.kv_cache_kernel_block_id_device, &dst.kv_cache_kernel_block_id_device};
    tables[1] = {&src.kv_cache_block_id_device, &dst.kv_cache_block_id_device};
    for (size_t g = 0; g < groups; ++g) {
        tables[g + 2] = {&src.kv_cache_kernel_block_id_device_by_group[g],
                         &dst.kv_cache_kernel_block_id_device_by_group[g]};
    }
    CudaGraphPrepareFillParams fills;
    for (size_t i = 0; i < groups + 2; ++i) {
        const auto& tensor = *tables[i].dst;
        if (!tensor.defined() || !tensor.is_cuda() || tensor.scalar_type() != torch::kInt32
            || tensor.dim() != 2 || !tensor.is_contiguous() || tensor.size(0) != capacity) {
            return false;
        }
        fills.regions[fills.region_count++] = {tensor.data_ptr<int32_t>(), tensor.numel(), 0};
    }
    CudaGraphCommitMetadataParams plan;
    if (!makeGraphTableOverwritePlan(tables, static_cast<int>(groups + 2), fills,
        {&src.input_lengths, &src.prefix_lengths, &src.sequence_lengths_plus_1_d,
         &dst.input_lengths, &dst.prefix_lengths, &dst.cu_seqlens, &dst.cu_kv_seqlens,
         &dst.sequence_lengths_plus_1_d, &dst.sequence_lengths, &dst.decode_cu_seqlens_d,
         &inputs.input_ids, &inputs.input_hiddens, &captured.input_ids, &captured.input_hiddens}, plan.tables)) {
        return false;
    }
    // Protect full storages for arbitrary views. Conservative rejection is
    // preferable to cross-CTA read/write hazards or accidental payload writes.
    const auto span = [](const torch::Tensor& tensor, GraphTableAddressRange& range) {
        range = {};
        if (!tensor.defined() || !tensor.is_cuda() || tensor.numel() == 0) return true;
        const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get());
        const size_t bytes = tensor.storage().nbytes();
        if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return false;
        range = {begin, begin + bytes};
        return true;
    };
    std::vector<const torch::Tensor*> written{&dst.input_lengths, &dst.prefix_lengths, &dst.cu_seqlens,
                                             &dst.cu_kv_seqlens, &dst.sequence_lengths_plus_1_d};
    if (capacity > b) {
        written.push_back(&dst.sequence_lengths);
        written.push_back(&dst.decode_cu_seqlens_d);
    }
    std::vector<const torch::Tensor*> protected_tensors{
        &src.input_lengths, &src.prefix_lengths, &src.sequence_lengths_plus_1_d,
        &inputs.input_ids, &inputs.input_hiddens, &captured.input_ids, &captured.input_hiddens,
        &dst.sequence_lengths, &dst.decode_cu_seqlens_d};
    for (size_t i = 0; i < groups + 2; ++i) {
        protected_tensors.push_back(tables[i].src);
        protected_tensors.push_back(tables[i].dst);
    }
    for (size_t i = 0; i < written.size(); ++i) {
        GraphTableAddressRange write;
        if (!span(*written[i], write)) return false;
        for (size_t j = 0; j < written.size(); ++j) {
            if (i == j) continue;
            GraphTableAddressRange other;
            if (!span(*written[j], other) || graphTableRangesOverlap(write, other)) return false;
        }
        for (const auto* tensor : protected_tensors) {
            // Optional tail writers own their own storage, not a live reader.
            if (tensor == written[i]) continue;
            GraphTableAddressRange other;
            if (!span(*tensor, other) || graphTableRangesOverlap(write, other)) return false;
        }
    }
    plan.input_lengths = src.input_lengths.data_ptr<int32_t>();
    plan.prefix_lengths = src.prefix_lengths.data_ptr<int32_t>();
    plan.sequence_lengths_plus_1 = src.sequence_lengths_plus_1_d.defined() ?
                                      src.sequence_lengths_plus_1_d.data_ptr<int32_t>() : nullptr;
    plan.output_input_lengths = dst.input_lengths.data_ptr<int32_t>();
    plan.output_prefix_lengths = dst.prefix_lengths.data_ptr<int32_t>();
    plan.output_cu_seqlens = dst.cu_seqlens.data_ptr<int32_t>();
    plan.output_cu_kv_seqlens = dst.cu_kv_seqlens.data_ptr<int32_t>();
    plan.output_sequence_lengths_plus_1 = dst.sequence_lengths_plus_1_d.data_ptr<int32_t>();
    if (capacity > b) {
        plan.output_sequence_lengths = dst.sequence_lengths.data_ptr<int32_t>();
        plan.output_decode_cu_seqlens = dst.decode_cu_seqlens_d.data_ptr<int32_t>();
    }
    plan.batch_size = b;
    plan.batch_capacity = capacity;
    plan.last_valid = state.seq_len_sum;
    if (plan.last_valid <= 0) return false;
    output = plan;
    return true;
}

} // namespace detail
} // namespace rtp_llm
