#pragma once

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

class SpecLogitsVerifyRunner {
public:
    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx      = 0;
        size_t                 processor_idx   = 0;
        uint64_t               stream_id       = 0;
        int64_t                base_seq_len    = 0;
        int64_t                base_output_len = 0;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor> active;
        size_t                       total_streams = 0;
        int                          propose_step  = 0;
        size_t                       vocab_size    = 0;

        // Shape [B, P], dtype int32/int64, CPU or CUDA.
        torch::Tensor                 draft_tokens;
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
    };

    struct LaunchResult {
        // Packed int32 allow mask [B * (P + 1), ceil(vocab_size / 32)].
        // Low bit first, 1 means allowed. Dense bool is retained for legacy
        // callers; CUDA buildInline produces compact, HIP retains dense bool.
        torch::Tensor                      spec_vocab_bitmask_gpu;
        torch::Tensor                      spec_vocab_mask_gpu;
        torch::Tensor                      spec_cap_gpu;  // [B] int32 CUDA
        std::shared_ptr<torch::Event>      ready_event;
        std::shared_ptr<torch::Event>      consumed_event;
        bool                               has_active_processor = false;
        std::vector<SpecLogitsProcessorId> applied_processors;

        // Keep H2D sources alive until ready_event has completed. Returned
        // views prevent pooled Storage reuse for as long as an owner or alias
        // survives, even after transfer completion; GPU outputs remain fresh.
        torch::Tensor spec_vocab_mask_cpu_owner;
        torch::Tensor spec_cap_cpu_owner;
        // Small immutable H2D descriptor source; fast results have no CPU bitmap.
        torch::Tensor static_think_descriptor_cpu_owner;
    };

    SpecLogitsVerifyRunner();

    LaunchResult buildInline(const LaunchTask& task);

private:
    std::optional<LaunchResult> tryBuildStaticThinkMask(const LaunchTask& task);
    void ensureBuffersFit(size_t total_streams, int propose_step, size_t vocab_size, size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);
#if USING_CUDA
    // Call only after the CUDA draft D2H has synchronized copy_stream_.
    std::pair<torch::Tensor, torch::Tensor> acquirePinnedStaging(int64_t rows, int64_t words, int64_t streams);
#endif

private:
    torch::Stream copy_stream_;
    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;
    torch::Tensor spec_cap_cpu_;
#if USING_CUDA
    struct HostStagingSlot {
        torch::Tensor mask;
        torch::Tensor cap;
    };
    // At most four slots, each mask+cap <=16 MiB (64 MiB per runner).
    // Result views keep their Storage shared and therefore immutable. Extra
    // outstanding or over-budget results use fresh staging.
    std::array<HostStagingSlot, 4> host_staging_;
#endif
};

}  // namespace rtp_llm
