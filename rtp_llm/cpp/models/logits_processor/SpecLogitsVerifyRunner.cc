#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>
#include <limits>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/compact_spec_vocab_mask.h"
#endif

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

void fillAllAllow(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
        std::fill_n(tensor.data_ptr<int32_t>(), tensor.numel(), SpecLogitsProcessor::kBitmaskAllowAll);
    }
}

void bitwiseAndInplace(int32_t* dst, const int32_t* src, size_t words) {
    for (size_t i = 0; i < words; ++i) {
        dst[i] &= src[i];
    }
}

#if !USING_CUDA
// HIP retains the original dense-bool sampler contract until it has a native
// packed-mask consumer. The CUDA path never executes this expansion.
void unpackMergedBitmaskToVocabMask(const torch::Tensor& merged,
                                    const torch::Tensor& mask,
                                    size_t rows,
                                    size_t vocab_size,
                                    size_t words) {
    const auto* bits = merged.data_ptr<int32_t>();
    auto* dense = mask.data_ptr<bool>();
    for (size_t row = 0; row < rows; ++row) {
        for (size_t token = 0; token < vocab_size; ++token) {
            const uint32_t word = static_cast<uint32_t>(bits[row * words + token / 32]);
            dense[row * vocab_size + token] = (word & (uint32_t{1} << (token % 32))) == 0;
        }
    }
}
#endif

}  // namespace

SpecLogitsVerifyRunner::SpecLogitsVerifyRunner(): copy_stream_(cuda_graph::graphGetStreamFromPool(true)) {

}

std::optional<SpecLogitsVerifyRunner::LaunchResult>
SpecLogitsVerifyRunner::tryBuildStaticThinkMask(const LaunchTask& task) {
#if USING_CUDA
    // Complete host preflight before device allocations or draft materialization.
    const size_t B = task.total_streams, V = task.vocab_size;
    const int P = task.propose_step;
    if (task.active.empty() || !B || B > INT32_MAX || P <= 0 || P == INT32_MAX || !V
        || V > static_cast<size_t>(INT64_MAX - 31) || !task.draft_tokens.defined()
        || !task.draft_tokens.is_cuda() || task.draft_tokens.device().index() != copy_stream_.device_index()
        || (task.draft_tokens.scalar_type() != torch::kInt32 && task.draft_tokens.scalar_type() != torch::kInt64))
        return std::nullopt;
    const size_t W = SpecLogitsProcessor::bitmaskWordCount(V);
    if (B > static_cast<size_t>(INT64_MAX) / (static_cast<size_t>(P) + 1)
        || B * (static_cast<size_t>(P) + 1) > static_cast<size_t>(INT64_MAX) / W)
        return std::nullopt;
    const int64_t elements = task.draft_tokens.numel();
    if (elements < static_cast<int64_t>(B) * P || elements % B != 0) return std::nullopt;
    const int64_t cols = elements / B, offset = cols > P ? 1 : 0;
    if (cols < offset + P) return std::nullopt;
    std::vector<StaticThinkMaskDescriptor> descriptors(B);
    std::vector<bool> active(B, false);
    for (const auto& item : task.active) {
        if (!item.processor || item.stream_idx >= B || active[item.stream_idx]
            || !item.processor->tryGetStaticThinkMaskDescriptor(descriptors[item.stream_idx])
            || !descriptors[item.stream_idx].snapshot_owner) return std::nullopt;
        active[item.stream_idx] = true;
    }

    LaunchResult result;
    auto pinned = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    auto host = torch::empty({static_cast<int64_t>(B), 3}, pinned);
    auto* values = host.data_ptr<int32_t>();
    for (size_t b = 0; b < B; ++b) {
        values[b * 3] = active[b];
        values[b * 3 + 1] = descriptors[b].first_begin_id;
        values[b * 3 + 2] = descriptors[b].first_end_id;
    }
    cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(copy_stream_));
    if (task.draft_tokens_ready_event) task.draft_tokens_ready_event->block(copy_stream_);
    auto draft = task.draft_tokens.reshape({static_cast<int64_t>(B), cols}).narrow(1, offset, P).contiguous();
    const auto stream = at::cuda::getCurrentCUDAStream(task.draft_tokens.device().index());
    c10::cuda::CUDACachingAllocator::recordStream(task.draft_tokens.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(draft.storage().data_ptr(), stream);
    auto options = task.draft_tokens.options().dtype(torch::kInt32);
    auto device_descriptors = host.to(options, /*non_blocking=*/true);
    result.spec_vocab_bitmask_gpu = torch::empty({static_cast<int64_t>(B) * (P + 1), static_cast<int64_t>(W)}, options);
    result.spec_cap_gpu = torch::empty({static_cast<int64_t>(B)}, options);
    buildStaticThinkSpecMask(draft, device_descriptors, result.spec_vocab_bitmask_gpu, result.spec_cap_gpu);
    result.ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    result.ready_event->record(copy_stream_);
    result.consumed_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    result.has_active_processor = true;
    result.static_think_descriptor_cpu_owner = std::move(host);
    for (const auto& item : task.active) result.applied_processors.push_back({item.stream_id, item.processor_idx});
    return result;
#else
    return std::nullopt;
#endif
}

void SpecLogitsVerifyRunner::ensureBuffersFit(size_t total_streams,
                                              int    propose_step,
                                              size_t vocab_size,
                                              size_t bitmask_words) {
    const int64_t B    = static_cast<int64_t>(total_streams);
    const int64_t P    = static_cast<int64_t>(propose_step);
    const int64_t rows = B * (P + 1);
    const int64_t W    = static_cast<int64_t>(bitmask_words);
    (void)vocab_size;

    auto cpu_i32     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32  = cpu_i32.pinned_memory(true);

    if (!draft_tokens_cpu_.defined() || draft_tokens_cpu_.numel() < B * P) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    if (!processor_bitmask_cpu_.defined() || processor_bitmask_cpu_.numel() < (P + 1) * W) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, cpu_i32);
    }
    if (!merged_bitmask_cpu_.defined() || merged_bitmask_cpu_.numel() < rows * W) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, cpu_i32);
    }
    if (!spec_cap_cpu_.defined() || spec_cap_cpu_.numel() < B) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
}

void SpecLogitsVerifyRunner::materializeDraftTokensToCpu(const LaunchTask& task) {
    const int64_t B = static_cast<int64_t>(task.total_streams);
    const int64_t P = static_cast<int64_t>(task.propose_step);
    if (B == 0 || P == 0) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.defined(), "spec logits runner requires draft tokens");
    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.numel() >= B * P && task.draft_tokens.numel() % B == 0,
                            "spec logits runner draft token shape mismatch");
    const int64_t draft_cols = task.draft_tokens.numel() / B;
    const int64_t draft_offset = draft_cols > P ? 1 : 0;
    RTP_LLM_CHECK_WITH_INFO(draft_cols >= draft_offset + P, "spec logits runner draft token columns mismatch");
    auto dst = draft_tokens_cpu_.reshape({-1}).narrow(0, 0, B * P).view({B, P});
    if (!task.draft_tokens.is_cuda()) {
        auto draft = task.draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
        auto draft_i32 =
            draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
        dst.copy_(draft_i32);
        return;
    }

    if (task.draft_tokens.device().index() != copy_stream_.device_index()) {
        // A wait on this runner's stream does not order operations on another
        // device's default stream. This rare legacy fallback uses a host fence
        // before any reshape/cast/read on the source device.
        if (task.draft_tokens_ready_event) task.draft_tokens_ready_event->synchronize();
        // Retain the prior H2D completion proof before pooled staging can be reused.
        copy_stream_.synchronize();
        c10::DeviceGuard source_guard(task.draft_tokens.device());
        auto draft = task.draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
        auto draft_i32 =
            draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
        dst.copy_(draft_i32.to(torch::kCPU));
        return;
    }

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    if (task.draft_tokens_ready_event) {
        task.draft_tokens_ready_event->block(copy_stream_);
    }
    // Reshape itself can enqueue a copy for a noncontiguous draft, so it too
    // belongs on the ordered copy stream after the producer wait.
    auto draft = task.draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
    auto draft_i32 = draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
    dst.copy_(draft_i32, /*non_blocking=*/true);
    copy_stream_.synchronize();
}

#if USING_CUDA
std::pair<torch::Tensor, torch::Tensor>
SpecLogitsVerifyRunner::acquirePinnedStaging(int64_t rows, int64_t words, int64_t streams) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    constexpr int64_t max_elements = (16 * 1024 * 1024) / sizeof(int32_t);
    // Division before multiplication avoids overflow in the pool-budget test.
    // Larger tasks keep the original fresh-allocation contract; they cannot
    // leave arbitrarily large pinned allocations retained in this runner.
    if (rows <= 0 || words <= 0 || streams <= 0 || streams > max_elements
        || rows > (max_elements - streams) / words) {
        return {torch::empty({rows, words}, options), torch::empty({streams}, options)};
    }
    const int64_t mask_elements = rows * words;
    for (auto& slot : host_staging_) {
        // storage() returns a const Storage&, so this check does not itself
        // increment the StorageImpl refcount. TensorImpl refcounts would miss
        // aliases: every returned narrow/view has its own TensorImpl but shares
        // this Storage. Both buffers must be free before either is overwritten.
        if ((slot.mask.defined() && slot.mask.storage().use_count() != 1)
            || (slot.cap.defined() && slot.cap.storage().use_count() != 1)) {
            continue;
        }
        const int64_t mask_capacity = std::max(mask_elements, slot.mask.defined() ? slot.mask.numel() : int64_t{0});
        const int64_t cap_capacity = std::max(streams, slot.cap.defined() ? slot.cap.numel() : int64_t{0});
        // Independent high-water marks may otherwise exceed the combined
        // per-slot budget even when this round's requested bytes fit.
        if (mask_capacity > max_elements - cap_capacity) {
            continue;
        }
        if (!slot.mask.defined() || slot.mask.numel() < mask_elements) {
            slot.mask = torch::empty({mask_elements}, options);
        }
        if (!slot.cap.defined() || slot.cap.numel() < streams) {
            slot.cap = torch::empty({streams}, options);
        }
        // Always return new views, including exact-capacity requests. Merely
        // returning the owning Tensor would not add a StorageImpl reference.
        return {slot.mask.narrow(0, 0, mask_elements).view({rows, words}), slot.cap.narrow(0, 0, streams)};
    }
    return {torch::empty({rows, words}, options), torch::empty({streams}, options)};
}
#endif

SpecLogitsVerifyRunner::LaunchResult SpecLogitsVerifyRunner::buildInline(const LaunchTask& task) {
    RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.build_inline");
    LaunchResult result;
    if (task.active.empty()) {
        return result;
    }

    const size_t B    = task.total_streams;
    const int    P    = task.propose_step;
    const size_t V    = task.vocab_size;
    const size_t W    = SpecLogitsProcessor::bitmaskWordCount(V);
    const size_t rows = B * static_cast<size_t>(P + 1);
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid spec logits runner task");

    if (auto fast = tryBuildStaticThinkMask(task)) return std::move(*fast);

    ensureBuffersFit(B, P, V, W);
    materializeDraftTokensToCpu(task);

    auto merged = merged_bitmask_cpu_.reshape({-1})
                      .narrow(0, 0, static_cast<int64_t>(rows * W))
                      .view({static_cast<int64_t>(rows), static_cast<int64_t>(W)});
    fillAllAllow(merged);
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

    auto proc_mask = processor_bitmask_cpu_.reshape({-1})
                         .narrow(0, 0, static_cast<int64_t>((P + 1) * W))
                         .view({P + 1, static_cast<int64_t>(W)});
    bool applied_processor = false;
    for (const auto& item : task.active) {
        if (!item.processor || !item.processor->isSpecVerifyEligible()) {
            return {};
        }
        RTP_LLM_CHECK_WITH_INFO(item.stream_idx < B, "spec logits runner processor stream index out of range");
        applied_processor = true;

        fillAllAllow(proc_mask);
        SpecLogitsProcessorRequest request;
        request.draft_tokens       = draft_tokens_cpu_.data_ptr<int32_t>() + item.stream_idx * P;
        request.propose_step       = P;
        request.bitmask_cpu_out    = proc_mask.data_ptr<int32_t>();
        request.bitmask_size_int32 = W;
        request.vocab_size         = V;
        request.stream_id          = item.stream_id;
        request.base_seq_len       = item.base_seq_len;
        request.base_output_len    = item.base_output_len;

        int cap;
        {
            RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.processor");
            cap = item.processor->tryAcceptAndFillBitmask(request);
        }
        cap     = std::max(0, std::min(cap, P));

        RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.merge");
        auto* merged_row = merged_bitmask_cpu_.data_ptr<int32_t>() + item.stream_idx * (P + 1) * W;
        bitwiseAndInplace(merged_row, proc_mask.data_ptr<int32_t>(), static_cast<size_t>(P + 1) * W);
        auto* cap_ptr            = spec_cap_cpu_.data_ptr<int32_t>();
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
        result.applied_processors.push_back({item.stream_id, item.processor_idx});
    }

    if (!applied_processor) {
        return {};
    }

    auto pinned_i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

#if USING_CUDA
    const int64_t mask_columns = static_cast<int64_t>(W);
    const auto mask_cpu_options = pinned_i32;
    const auto mask_gpu_options = cuda_i32;
#else
    const int64_t mask_columns = static_cast<int64_t>(V);
    const auto mask_cpu_options = pinned_i32.dtype(torch::kBool);
    const auto mask_gpu_options = cuda_i32.dtype(torch::kBool);
#endif
    torch::Tensor mask_cpu, cap_cpu, mask_gpu, cap_gpu;
    {
        RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.allocate");
#if USING_CUDA
        // materializeDraftTokensToCpu synchronizes this same stream after its
        // GPU draft D2H. This also completes all earlier mask/cap H2D reads;
        // no event query or additional wait is needed before reusing staging.
        // CPU drafts have no such completion proof, so retain fresh ownership.
        if (task.draft_tokens.is_cuda()) {
            auto staging = acquirePinnedStaging(static_cast<int64_t>(rows), mask_columns, static_cast<int64_t>(B));
            mask_cpu = std::move(staging.first);
            cap_cpu = std::move(staging.second);
        } else
#endif
        {
            mask_cpu = torch::empty({static_cast<int64_t>(rows), mask_columns}, mask_cpu_options);
            cap_cpu  = torch::empty({static_cast<int64_t>(B)}, pinned_i32);
        }
        mask_gpu = torch::empty({static_cast<int64_t>(rows), mask_columns}, mask_gpu_options);
        cap_gpu  = torch::empty({static_cast<int64_t>(B)}, cuda_i32);
    }
    {
        RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.pack_copy");
#if USING_CUDA
        mask_cpu.copy_(merged);
#else
        unpackMergedBitmaskToVocabMask(merged, mask_cpu, rows, V, W);
#endif
        cap_cpu.copy_(spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B)));
    }

    RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.h2d_submit");
    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    mask_gpu.copy_(mask_cpu, /*non_blocking=*/true);
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);
    auto ready = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    ready->record(copy_stream_);
    auto consumed = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());

#if USING_CUDA
    result.spec_vocab_bitmask_gpu = mask_gpu;
#else
    result.spec_vocab_mask_gpu = mask_gpu;
#endif
    result.spec_cap_gpu         = cap_gpu;
    result.ready_event          = ready;
    result.consumed_event       = consumed;
    result.has_active_processor = true;
    result.spec_vocab_mask_cpu_owner = mask_cpu;
    result.spec_cap_cpu_owner        = cap_cpu;
    return result;
}

}  // namespace rtp_llm
