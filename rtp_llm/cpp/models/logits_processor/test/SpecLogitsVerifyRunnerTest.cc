#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

namespace rtp_llm {
namespace {

bool isMasked(const torch::Tensor& mask, int64_t row, int64_t token, int64_t vocab) {
    if (mask.scalar_type() == torch::kBool) {
        return mask.data_ptr<bool>()[row * vocab + token];
    }
    const auto word = static_cast<uint32_t>(mask.data_ptr<int32_t>()[row * ((vocab + 31) / 32) + token / 32]);
    return (word & (uint32_t{1} << (token % 32))) == 0;
}

std::string tokenizerInfo() {
    std::vector<std::string> vocab;
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    return xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0}).SerializeJSON();
}

// Fixed-budget runners match production: changing k on a live runner is not
// exercised here. Both requests have distinct, position-dependent masks.
class SpecLogitsVerifyRunnerTest: public ::testing::TestWithParam<int> {};

TEST_P(SpecLogitsVerifyRunnerTest, RealGrammarCapsAnchorAndAsyncOwnership) {
    const int                        k        = GetParam();
    constexpr int                    B        = 2;
    constexpr int                    V        = 160;  // Also cover tokens outside the grammar vocabulary.
    const std::array<std::string, B> patterns = {"abcdefgh", "mnopqrst"};
    XGrammarBackendOptions           options;
    options.max_compiler_threads = 1;
    XGrammarBackendCpp backend(tokenizerInfo(), options);
    const auto         pinned_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);

    const auto         cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto               producer    = cuda_graph::graphGetStreamFromPool(false);
    auto               consumer    = cuda_graph::graphGetStreamFromPool(false);

    for (bool with_anchor : {false, true}) {
        for (const auto& caps : std::vector<std::array<int, B>>{{0, k}, {k - 1, 0}, {k, k - 1}}) {
            SCOPED_TRACE(::testing::Message()
                         << "k=" << k << " anchor=" << with_anchor << " caps=" << caps[0] << "," << caps[1]);
            SpecLogitsVerifyRunner             runner;
            SpecLogitsVerifyRunner::LaunchTask task;
            task.total_streams = B;
            task.propose_step  = k;
            task.vocab_size    = V;
            std::array<std::shared_ptr<GrammarLogitsProcessor>, B> processors;
            for (int b = 0; b < B; ++b) {
                auto compiled = backend.compileNow({"regex", patterns[b].substr(0, k + 1)}).compiled;
                ASSERT_TRUE(compiled);
                processors[b] =
                    std::make_shared<GrammarLogitsProcessor>(backend.createMatcher(compiled, false, std::nullopt), 0);
                SpecLogitsVerifyRunner::ActiveProcessor active;
                active.processor     = processors[b];
                active.stream_idx    = b;
                active.stream_id     = 100 + b;
                active.processor_idx = b;
                task.active.push_back(active);
            }

            torch::Tensor first_mask;
            torch::Tensor first_cap;
            for (int round = 0; round < 2; ++round) {
                SCOPED_TRACE(::testing::Message() << "round=" << round);
                const int cols   = k + static_cast<int>(with_anchor);
                auto      source = torch::empty({B, cols}, pinned_i32);
                auto*     tokens = source.data_ptr<int32_t>();
                for (int b = 0; b < B; ++b) {
                    if (with_anchor) {
                        tokens[b * cols] = '@';  // Invalid: cap would become zero if not stripped.
                    }
                    for (int p = 0; p < k; ++p) {
                        tokens[b * cols + static_cast<int>(with_anchor) + p] = p == caps[b] ? '#' : patterns[b][p];
                    }
                }
                task.draft_tokens             = torch::empty({B, cols}, cuda_i32);
                task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
                {
                    cuda_graph::GraphStreamGuard guard(producer);
                    task.draft_tokens.copy_(source, /*non_blocking=*/true);
                    task.draft_tokens_ready_event->record(producer);
                }
                auto result = runner.buildInline(task);
                ASSERT_TRUE(result.has_active_processor);
#if USING_CUDA
                ASSERT_FALSE(result.spec_vocab_mask_gpu.defined());
                ASSERT_EQ(result.spec_vocab_bitmask_gpu.scalar_type(), torch::kInt32);
                auto gpu_mask = result.spec_vocab_bitmask_gpu;
                ASSERT_EQ(gpu_mask.size(1), (V + 31) / 32);
#else
                ASSERT_FALSE(result.spec_vocab_bitmask_gpu.defined());
                ASSERT_EQ(result.spec_vocab_mask_gpu.scalar_type(), torch::kBool);
                auto gpu_mask = result.spec_vocab_mask_gpu;
                ASSERT_EQ(gpu_mask.size(1), V);
#endif
                ASSERT_EQ(result.applied_processors.size(), B);
                ASSERT_TRUE(result.ready_event);
                ASSERT_TRUE(result.consumed_event);
                ASSERT_TRUE(result.spec_vocab_mask_cpu_owner.defined());
                ASSERT_TRUE(result.spec_cap_cpu_owner.defined());
                ASSERT_EQ(gpu_mask.size(0), B * (k + 1));
                ASSERT_EQ(result.spec_cap_gpu.numel(), B);

                auto mask = torch::empty(gpu_mask.sizes(), gpu_mask.options().device(torch::kCPU).pinned_memory(true));
                auto cap  = torch::empty({B}, pinned_i32);
                {
                    cuda_graph::GraphStreamGuard guard(consumer);
                    result.ready_event->block(consumer);
                    mask.copy_(gpu_mask, /*non_blocking=*/true);
                    cap.copy_(result.spec_cap_gpu, /*non_blocking=*/true);
                    result.consumed_event->record(consumer);
                }
                // Keep result (including pinned H2D owners) and source alive
                // until the consumer's copies have completed. No global sync.
                result.consumed_event->synchronize();
                for (int b = 0; b < B; ++b) {
                    EXPECT_EQ(cap.data_ptr<int32_t>()[b], caps[b]);
                    EXPECT_EQ(processors[b]->acceptedTokenLen(), 0);
                    for (int p = 0; p <= k; ++p) {
                        for (int token = 0; token < V; ++token) {
                            // Rows beyond the rejection cap are unvisited and
                            // remain all-allow; row cap itself must be valid.
                            const bool expected_mask = p <= caps[b] && token != patterns[b][p];
                            EXPECT_EQ(isMasked(mask, b * (k + 1) + p, token, V), expected_mask)
                                << "batch=" << b << " offset=" << p << " token=" << token;
                        }
                    }
                }
                if (round == 0) {
                    first_mask = mask;
                    first_cap  = cap;
                } else {
                    EXPECT_TRUE(torch::equal(first_mask, mask));
                    EXPECT_TRUE(torch::equal(first_cap, cap));
                }
            }
        }
    }
}

class PatternProcessor: public SpecLogitsProcessor {
public:
    PatternProcessor(int modulus, int cap): modulus_(modulus), cap_(cap) {}
    bool isSpecVerifyEligible() const override { return true; }
    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        for (int p = 0; p < request.propose_step; ++p) {
            EXPECT_EQ(request.draft_tokens[p], p + 10);
        }
        std::fill_n(request.bitmask_cpu_out, (request.propose_step + 1) * request.bitmask_size_int32, 0);
        for (int p = 0; p <= request.propose_step; ++p) {
            for (size_t token = 0; token < request.vocab_size; ++token) {
                if (token % modulus_ == static_cast<size_t>(p % modulus_)) {
                    auto& word = request.bitmask_cpu_out[p * request.bitmask_size_int32 + token / 32];
                    word = static_cast<int32_t>(static_cast<uint32_t>(word) | (uint32_t{1} << (token % 32)));
                }
            }
        }
        return cap_;
    }
private:
    int modulus_, cap_;
};

TEST(SpecLogitsVerifyRunnerStandaloneTest, MergeRaggedShapeAndOutstandingOwners) {
    SpecLogitsVerifyRunner runner;
    std::vector<SpecLogitsVerifyRunner::LaunchResult> results;
    // Shrink both P and W on the same scratch owner, then grow again. Queue
    // every copy before consuming any result to catch premature reuse.
    const std::vector<std::pair<int, int>> shapes = {{7, 161}, {1, 33}, {4, 65}, {7, 200064}};
    for (const auto& [p, vocab] : shapes) {
        SpecLogitsVerifyRunner::LaunchTask task;
        task.total_streams = 2;
        task.propose_step = p;
        task.vocab_size = vocab;
        task.draft_tokens = torch::arange(10, 10 + p, torch::kInt64).repeat({2, 1});
        auto a = std::make_shared<PatternProcessor>(2, p);
        auto b = std::make_shared<PatternProcessor>(3, p - 1);
        task.active = {{a, 0, 0, 123}, {b, 0, 1, 123}};
        results.push_back(runner.buildInline(task));
    }
    for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];
        const auto [p, vocab] = shapes[i];
        ASSERT_EQ(result.applied_processors.size(), 2);
        EXPECT_EQ(result.applied_processors[0].stream_id, 123);
        EXPECT_EQ(result.applied_processors[1].processor_idx, 1);
        result.ready_event->block(cuda_graph::graphGetCurrentStream());
#if USING_CUDA
        ASSERT_FALSE(result.spec_vocab_mask_gpu.defined());
        auto mask = result.spec_vocab_bitmask_gpu.to(torch::kCPU);
#else
        ASSERT_FALSE(result.spec_vocab_bitmask_gpu.defined());
        auto mask = result.spec_vocab_mask_gpu.to(torch::kCPU);
#endif
        auto caps = result.spec_cap_gpu.to(torch::kCPU);
        result.consumed_event->record(cuda_graph::graphGetCurrentStream());
        result.consumed_event->synchronize();
        EXPECT_EQ(caps.data_ptr<int32_t>()[0], p - 1);
        EXPECT_EQ(caps.data_ptr<int32_t>()[1], p);
        for (int row = 0; row < 2 * (p + 1); ++row) {
            for (int token = 0; token < vocab; ++token) {
                const bool expected_allow = row > p || (token % 2 == row % 2 && token % 3 == row % 3);
                EXPECT_EQ(!isMasked(mask, row, token, vocab), expected_allow);
            }
        }
    }
}

#if USING_CUDA
SpecLogitsVerifyRunner::LaunchTask makeStagingTask(int streams, int p, int vocab, int modulus, bool gpu = true) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = streams;
    task.propose_step = p;
    task.vocab_size = vocab;
    task.draft_tokens = torch::arange(10, 10 + p, torch::kInt32).repeat({streams, 1});
    if (gpu) {
        task.draft_tokens = task.draft_tokens.to(torch::kCUDA);
        task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        task.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
    }
    for (int stream = 0; stream < streams; ++stream) {
        task.active.push_back({std::make_shared<PatternProcessor>(modulus, p - 1),
                               static_cast<size_t>(stream), 0, static_cast<uint64_t>(1000 + stream)});
    }
    return task;
}

void checkStagingResult(SpecLogitsVerifyRunner::LaunchResult& result) {
    ASSERT_TRUE(result.has_active_processor);
    result.ready_event->block(cuda_graph::graphGetCurrentStream());
    EXPECT_TRUE(torch::equal(result.spec_vocab_bitmask_gpu.to(torch::kCPU), result.spec_vocab_mask_cpu_owner));
    EXPECT_TRUE(torch::equal(result.spec_cap_gpu.to(torch::kCPU), result.spec_cap_cpu_owner));
    result.consumed_event->record(cuda_graph::graphGetCurrentStream());
    result.consumed_event->synchronize();
}

TEST(SpecLogitsVerifyRunnerStagingTest, RetainedOwnersAndAliasesStayImmutableWithBoundedPool) {
    SpecLogitsVerifyRunner runner;
    std::vector<SpecLogitsVerifyRunner::LaunchResult> held;
    std::vector<torch::Tensor> masks, caps;
    // Four retained slots plus two fresh fallbacks. Every round changes the
    // mask contents so an overwritten owner is observable, not just an alias.
    for (int i = 0; i < 6; ++i) {
        auto task = makeStagingTask(2, 4, 161, i + 2);
        held.push_back(runner.buildInline(task));
        masks.push_back(held.back().spec_vocab_mask_cpu_owner.clone());
        caps.push_back(held.back().spec_cap_cpu_owner.clone());
        for (int j = 0; j < i; ++j) {
            EXPECT_NE(held[j].spec_vocab_mask_cpu_owner.data_ptr(), held[i].spec_vocab_mask_cpu_owner.data_ptr());
            EXPECT_NE(held[j].spec_cap_cpu_owner.data_ptr(), held[i].spec_cap_cpu_owner.data_ptr());
            EXPECT_NE(held[j].spec_vocab_bitmask_gpu.data_ptr(), held[i].spec_vocab_bitmask_gpu.data_ptr());
            EXPECT_NE(held[j].spec_cap_gpu.data_ptr(), held[i].spec_cap_gpu.data_ptr());
        }
    }
    for (size_t i = 0; i < held.size(); ++i) {
        EXPECT_TRUE(torch::equal(held[i].spec_vocab_mask_cpu_owner, masks[i]));
        EXPECT_TRUE(torch::equal(held[i].spec_cap_cpu_owner, caps[i]));
        checkStagingResult(held[i]);
    }
    // Drop all but a distinct view/alias of the first mask. Holding just mask
    // must exclude its slot as a pair, even when the cap owner is released.
    auto alias = held[0].spec_vocab_mask_cpu_owner.narrow(0, 0, 1);
    auto alias_snapshot = alias.clone();
    const auto first_address = held[0].spec_vocab_mask_cpu_owner.data_ptr();
    held.clear();
    auto task = makeStagingTask(2, 4, 161, 11);
    auto next = runner.buildInline(task);
    EXPECT_NE(next.spec_vocab_mask_cpu_owner.data_ptr(), first_address);
    EXPECT_TRUE(torch::equal(alias, alias_snapshot));
    checkStagingResult(next);
}

TEST(SpecLogitsVerifyRunnerStagingTest, SequentialReleaseReusesStorageAndChangedShapeRefreshesEveryRow) {
    SpecLogitsVerifyRunner runner;
    auto first = runner.buildInline(makeStagingTask(3, 7, 161, 2));
    checkStagingResult(first);
    const auto mask_address = first.spec_vocab_mask_cpu_owner.data_ptr();
    const auto cap_address = first.spec_cap_cpu_owner.data_ptr();
    first = SpecLogitsVerifyRunner::LaunchResult();
    // Change cohort, both strides and contents while staying within capacity.
    auto small = runner.buildInline(makeStagingTask(1, 1, 33, 5));
    EXPECT_EQ(small.spec_vocab_mask_cpu_owner.data_ptr(), mask_address);
    EXPECT_EQ(small.spec_cap_cpu_owner.data_ptr(), cap_address);
    EXPECT_EQ(small.spec_vocab_mask_cpu_owner.size(0), 2);
    EXPECT_EQ(small.spec_vocab_mask_cpu_owner.size(1), 2);
    EXPECT_EQ(small.spec_cap_cpu_owner.numel(), 1);
    EXPECT_EQ(small.spec_cap_cpu_owner.data_ptr<int32_t>()[0], 0);
    for (int row = 0; row < 2; ++row) {
        for (int token = 0; token < 33; ++token) {
            EXPECT_EQ(!isMasked(small.spec_vocab_mask_cpu_owner, row, token, 33), token % 5 == row % 5);
        }
    }
    checkStagingResult(small);
    small = SpecLogitsVerifyRunner::LaunchResult();
    // Growing an unused slot is allowed. No old owner survives to observe it.
    auto large = runner.buildInline(makeStagingTask(4, 7, 200064, 3));
    EXPECT_EQ(large.spec_vocab_mask_cpu_owner.size(0), 32);
    EXPECT_EQ(large.spec_vocab_mask_cpu_owner.size(1), (200064 + 31) / 32);
    checkStagingResult(large);
}

TEST(SpecLogitsVerifyRunnerStagingTest, CpuDraftAlwaysUsesFreshOwnersAndDoesNotBorrowPool) {
    SpecLogitsVerifyRunner runner;
    auto gpu = runner.buildInline(makeStagingTask(2, 4, 161, 2));
    checkStagingResult(gpu);
    const auto pool_address = gpu.spec_vocab_mask_cpu_owner.data_ptr();
    gpu = SpecLogitsVerifyRunner::LaunchResult();
    auto a = runner.buildInline(makeStagingTask(2, 4, 161, 3, false));
    auto snapshot = a.spec_vocab_mask_cpu_owner.clone();
    auto b = runner.buildInline(makeStagingTask(2, 4, 161, 5, false));
    EXPECT_NE(a.spec_vocab_mask_cpu_owner.data_ptr(), pool_address);
    EXPECT_NE(b.spec_vocab_mask_cpu_owner.data_ptr(), pool_address);
    EXPECT_NE(a.spec_vocab_mask_cpu_owner.data_ptr(), b.spec_vocab_mask_cpu_owner.data_ptr());
    EXPECT_NE(a.spec_cap_cpu_owner.data_ptr(), b.spec_cap_cpu_owner.data_ptr());
    EXPECT_TRUE(torch::equal(a.spec_vocab_mask_cpu_owner, snapshot));
    checkStagingResult(a);
    checkStagingResult(b);
    auto c = runner.buildInline(makeStagingTask(2, 4, 161, 7));
    EXPECT_EQ(c.spec_vocab_mask_cpu_owner.data_ptr(), pool_address);
    checkStagingResult(c);
}

TEST(SpecLogitsVerifyRunnerStagingTest, CombinedHighWaterCapacityAndOverBudgetTasksStayBounded) {
    SpecLogitsVerifyRunner runner;
    constexpr int64_t max_elements = (16 * 1024 * 1024) / sizeof(int32_t);
    // Private helper is host-only; no transfers use these sources. One cap
    // element takes the requested pair just over the 16 MiB combined budget.
    auto outside = runner.acquirePinnedStaging(1, max_elements, 1);
    for (const auto& slot : runner.host_staging_) {
        EXPECT_FALSE(slot.mask.defined());
        EXPECT_FALSE(slot.cap.defined());
    }
    outside = std::pair<torch::Tensor, torch::Tensor>();
    auto mask_heavy = runner.acquirePinnedStaging(1, 3 * max_elements / 4, 1);
    const auto mask_address = mask_heavy.first.data_ptr();
    mask_heavy = std::pair<torch::Tensor, torch::Tensor>();
    // Both individual rounds fit, but slot zero's independently grown mask
    // and cap would exceed the budget. Choose another unused slot instead.
    auto cap_heavy = runner.acquirePinnedStaging(1, 1, max_elements / 2);
    EXPECT_NE(cap_heavy.first.data_ptr(), mask_address);
    int64_t retained = 0;
    for (const auto& slot : runner.host_staging_) {
        const auto elements = (slot.mask.defined() ? slot.mask.numel() : 0)
                              + (slot.cap.defined() ? slot.cap.numel() : 0);
        EXPECT_LE(elements, max_elements);
        retained += elements;
    }
    EXPECT_LE(retained, 4 * max_elements);
}

class DelayedCopyPatternProcessor: public PatternProcessor {
public:
    DelayedCopyPatternProcessor(cudaStream_t stream, std::shared_ptr<std::atomic<bool>> done):
        PatternProcessor(2, 3), stream_(stream), done_(std::move(done)) {}
    int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override {
        const int cap = PatternProcessor::tryAcceptAndFillBitmask(request);
        // The first draft D2H already synchronized; delay this result's H2D
        // reads to test the NEXT draft D2H completion proof before reuse.
        auto* context = new std::shared_ptr<std::atomic<bool>>(done_);
        const auto error = cudaLaunchHostFunc(stream_, [](void* data) {
            std::unique_ptr<std::shared_ptr<std::atomic<bool>>> owner(
                static_cast<std::shared_ptr<std::atomic<bool>>*>(data));
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            (*owner)->store(true, std::memory_order_release);
        }, context);
        if (error != cudaSuccess) {
            delete context;
        }
        EXPECT_EQ(error, cudaSuccess);
        return cap;
    }
private:
    cudaStream_t stream_;
    std::shared_ptr<std::atomic<bool>> done_;
};

TEST(SpecLogitsVerifyRunnerStagingTest, NextGpuDraftSynchronizeCompletesDelayedPreviousH2DWithoutResultOwner) {
    SpecLogitsVerifyRunner runner;
    auto done = std::make_shared<std::atomic<bool>>(false);
    auto task = makeStagingTask(1, 4, 200064, 2);
    // Prepare the next GPU inputs before the delay, so its creation cannot
    // accidentally complete the first upload through an allocator/copy wait.
    auto next_task = makeStagingTask(1, 4, 200064, 3);
    // This target is compiled with -fno-access-control. Access the actual
    // runner copy stream to queue a deterministic delay; no production hook.
    task.active[0].processor = std::make_shared<DelayedCopyPatternProcessor>(
        cuda_graph::toGraphStream(runner.copy_stream_).stream(), done);
    auto first = runner.buildInline(task);
    auto expected = first.spec_vocab_mask_cpu_owner.clone();
    auto expected_cap = first.spec_cap_cpu_owner.clone();
    auto first_gpu = first.spec_vocab_bitmask_gpu;
    auto first_cap_gpu = first.spec_cap_gpu;
    auto ready = first.ready_event;
    const auto host_address = first.spec_vocab_mask_cpu_owner.data_ptr();
    // Release the CPU owner before its delayed upload. The pool retains it.
    first = SpecLogitsVerifyRunner::LaunchResult();
    auto next = runner.buildInline(next_task);
    EXPECT_TRUE(done->load(std::memory_order_acquire));
    EXPECT_EQ(next.spec_vocab_mask_cpu_owner.data_ptr(), host_address);
    ready->block(cuda_graph::graphGetCurrentStream());
    EXPECT_TRUE(torch::equal(first_gpu.to(torch::kCPU), expected));
    EXPECT_TRUE(torch::equal(first_cap_gpu.to(torch::kCPU), expected_cap));
    checkStagingResult(next);
}

#endif


#if USING_CUDA
std::shared_ptr<ThinkModeLogitsProcessor> stableThink(ThinkProcessState state,
                                                   std::vector<int> begin = {2, 7},
                                                   std::vector<int> end = {3, 8}) {
    StreamThinkInfo info(ThinkingMode::DISABLED, 0, std::move(begin), std::move(end), 1, 0, false, nullptr);
    info.process_state = state;
    return std::make_shared<ThinkModeLogitsProcessor>(std::vector<StreamThinkInfo>{info});
}

SpecLogitsVerifyRunner::LaunchTask stableTask(const torch::Tensor& draft,
                                            std::shared_ptr<ThinkModeLogitsProcessor> processor,
                                            int p, int vocab = 33) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = 2;
    task.propose_step = p;
    task.vocab_size = vocab;
    task.draft_tokens = draft;
    // Stream 1 deliberately has no processor: cap P and every word all-allowed.
    task.active = {{processor, 0, 9, 1234}};
    if (draft.is_cuda()) {
        task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        task.draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
    }
    return task;
}

void expectStaticResult(SpecLogitsVerifyRunner::LaunchResult& actual,
                        const SpecLogitsVerifyRunner::LaunchResult& legacy) {
    ASSERT_TRUE(actual.static_think_descriptor_cpu_owner.defined());
    ASSERT_FALSE(actual.spec_vocab_mask_cpu_owner.defined());
    ASSERT_FALSE(actual.spec_cap_cpu_owner.defined());
    ASSERT_TRUE(actual.ready_event && actual.consumed_event);
    actual.ready_event->block(cuda_graph::graphGetCurrentStream());
    EXPECT_TRUE(torch::equal(actual.spec_vocab_bitmask_gpu.cpu(), legacy.spec_vocab_mask_cpu_owner));
    EXPECT_TRUE(torch::equal(actual.spec_cap_gpu.cpu(), legacy.spec_cap_cpu_owner));
    ASSERT_EQ(actual.applied_processors.size(), 1);
    EXPECT_EQ(actual.applied_processors[0].stream_id, 1234);
    EXPECT_EQ(actual.applied_processors[0].processor_idx, 9);
    actual.consumed_event->record(cuda_graph::graphGetCurrentStream());
}

TEST(StaticThinkSpecMaskTest, ActualRunnerMatchesCpuSnapshotEveryCapAnchorAndWordBoundary) {
    for (auto state : {ThinkProcessState::NO_THINK, ThinkProcessState::AFTER_THINK}) {
        for (auto dtype : {torch::kInt32, torch::kInt64}) {
            for (int p : {1, 4, 7}) {
                for (bool anchor : {false, true}) {
                    for (int blocked = -1; blocked < p; ++blocked) {
                        SCOPED_TRACE(::testing::Message() << "state=" << int(state) << " p=" << p
                                     << " anchor=" << anchor << " blocked=" << blocked);
                        auto processor = stableThink(state);
                        auto draft = torch::full({2, p + int(anchor)}, 7, torch::TensorOptions().dtype(dtype));
                        // Only the FIRST boundary ids block; secondary begin/end ids remain allowed.
                        if (anchor) draft.select(1, 0).fill_(2);
                        if (blocked >= 0) draft.index_put_({0, blocked + int(anchor)}, 2);
                        SpecLogitsVerifyRunner reference, runner;
                        const auto before = processor->acceptedTokenLen();
                        auto cpu = reference.buildInline(stableTask(draft, processor, p));
                        auto gpu = runner.buildInline(stableTask(draft.cuda(), processor, p));
                        expectStaticResult(gpu, cpu);
                        EXPECT_EQ(processor->acceptedTokenLen(), before);
                        // Eligible task did not allocate legacy scratch or acquire pooled staging.
                        EXPECT_FALSE(runner.draft_tokens_cpu_.defined());
                        EXPECT_FALSE(runner.merged_bitmask_cpu_.defined());
                    }
                }
            }
        }
    }
    {
        // Duplicate boundary ids are idempotent; empty boundaries preserve all bits.
        for (bool empty : {false, true}) {
            auto processor = stableThink(ThinkProcessState::AFTER_THINK,
                                         empty ? std::vector<int>{} : std::vector<int>{2, 2},
                                         empty ? std::vector<int>{} : std::vector<int>{2, 2});
            auto draft = torch::full({2, 4}, 2, torch::kInt32);
            SpecLogitsVerifyRunner reference, runner;
            auto cpu = reference.buildInline(stableTask(draft, processor, 4));
            auto gpu = runner.buildInline(stableTask(draft.cuda(), processor, 4));
            expectStaticResult(gpu, cpu);
        }
    }
    {
        // Noncontiguous GPU payload and an anchor still use the legacy column contract.
        auto base = torch::full({2, 10}, 10, torch::kInt64).cuda();
        auto strided = base.slice(1, 0, 10, 2);
        strided.index_put_({0, 0}, 2);
        strided.index_put_({0, 3}, 3);
        auto processor = stableThink(ThinkProcessState::NO_THINK);
        SpecLogitsVerifyRunner reference, runner;
        auto cpu = reference.buildInline(stableTask(strided.cpu(), processor, 4));
        auto gpu = runner.buildInline(stableTask(strided, processor, 4));
        expectStaticResult(gpu, cpu);
    }
    {
        // Gamma 7 plus anchor, while target verifies only the first P=4 columns.
        // Both the blocked anchor and blocked unverified tail must be ignored.
        for (int blocked : {-1, 2}) {
            auto draft = torch::full({2, 8}, 10, torch::kInt64);
            draft.select(1, 0).fill_(2);
            draft.slice(1, 5, 8).fill_(3);
            if (blocked >= 0) draft.index_put_({0, blocked + 1}, 2);
            auto processor = stableThink(ThinkProcessState::NO_THINK);
            SpecLogitsVerifyRunner reference, runner;
            auto cpu = reference.buildInline(stableTask(draft, processor, 4));
            auto gpu = runner.buildInline(stableTask(draft.cuda(), processor, 4));
            expectStaticResult(gpu, cpu);
            EXPECT_EQ(cpu.spec_cap_cpu_owner.data_ptr<int32_t>()[0], blocked < 0 ? 4 : blocked);
        }
    }
    for (int64_t token : {int64_t{-1}, int64_t{32}, int64_t{63}, int64_t{64}, int64_t{4294967298}}) {
        // V=33 has two packed words: token 63 is valid to the CPU bitmask contract.
        auto processor = stableThink(ThinkProcessState::NO_THINK, {63}, {3});
        auto draft = torch::full({2, 4}, 10, torch::kInt64);
        draft.index_put_({0, 2}, token);
        SpecLogitsVerifyRunner reference, runner;
        auto cpu = reference.buildInline(stableTask(draft, processor, 4));
        auto gpu = runner.buildInline(stableTask(draft.cuda(), processor, 4));
        expectStaticResult(gpu, cpu);
    }
}

TEST(StaticThinkSpecMaskTest, SnapshotRetainedAndUnsupportedWholeTaskFallsBack) {
    auto processor = stableThink(ThinkProcessState::NO_THINK);
    StaticThinkMaskDescriptor original, current;
    ASSERT_TRUE(processor->tryGetStaticThinkMaskDescriptor(original));
    processor->updateStatus(torch::tensor({{10}}, torch::kInt32), 1);
    ASSERT_TRUE(processor->tryGetStaticThinkMaskDescriptor(current));
    EXPECT_GT(current.snapshot_version, original.snapshot_version);
    EXPECT_EQ(std::static_pointer_cast<const ThinkModeSpecSnapshot>(original.snapshot_owner)->info.current_output_length, 0);
    // Publish a new state, retaining the exact old immutable descriptor snapshot.
    processor->think_infos_[0].begin_think_token_ids[0] = 12;
    processor->think_infos_[0].process_state = ThinkProcessState::AFTER_THINK;
    processor->publishSpecSnapshotLocked();
    ASSERT_TRUE(processor->tryGetStaticThinkMaskDescriptor(current));
    EXPECT_GT(current.snapshot_version, original.snapshot_version);
    EXPECT_NE(current.snapshot_owner.get(), original.snapshot_owner.get());
    EXPECT_EQ(original.first_begin_id, 2);
    EXPECT_EQ(current.first_begin_id, 12);
    auto retained = std::static_pointer_cast<const ThinkModeSpecSnapshot>(original.snapshot_owner);
    EXPECT_EQ(retained->info.begin_think_token_ids[0], 2);
    auto draft = torch::arange(10, 14, torch::kInt64).repeat({2, 1}).cuda();
    SpecLogitsVerifyRunner runner;
    for (int kind = 0; kind < 5; ++kind) {
        auto eligible = stableThink(ThinkProcessState::NO_THINK);
        auto task = stableTask(draft, eligible, 4);
        if (kind == 0) task.active.push_back({eligible, 0, 10, 1234}); // Duplicate stream.
        if (kind == 1) task.active.push_back({std::make_shared<PatternProcessor>(2, 3), 1, 1, 5678});
        if (kind == 2) eligible->think_infos_[0].pending_forced_think_end_token_ids = {3};
        if (kind == 3) eligible->think_infos_[0].process_state = ThinkProcessState::IN_THINK;
        if (kind == 4) eligible->think_infos_[0].is_beam_search = true;
        if (kind >= 2) eligible->publishSpecSnapshotLocked();
        EXPECT_FALSE(runner.tryBuildStaticThinkMask(task).has_value());
        auto fallback = runner.buildInline(task);
        EXPECT_FALSE(fallback.static_think_descriptor_cpu_owner.defined());
        if (kind == 4) {
            EXPECT_FALSE(fallback.has_active_processor); // Legacy rejects beam-ineligible processor unchanged.
        } else {
            EXPECT_TRUE(fallback.spec_vocab_mask_cpu_owner.defined());
            fallback.ready_event->block(cuda_graph::graphGetCurrentStream());
            fallback.spec_cap_gpu.cpu();
        }
    }
    // Actual grammar mixed with a stable Think processor must use the whole legacy task.
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    XGrammarBackendCpp backend(tokenizerInfo(), options);
    auto compiled = backend.compileNow({"regex", "abcdefgh"}).compiled;
    ASSERT_TRUE(compiled);
    auto grammar = std::make_shared<GrammarLogitsProcessor>(
        backend.createMatcher(compiled, false, std::nullopt), 0);
    auto mixed = stableTask(draft, stableThink(ThinkProcessState::NO_THINK), 4, 160);
    mixed.active.push_back({grammar, 1, 1, 5678});
    EXPECT_FALSE(runner.tryBuildStaticThinkMask(mixed).has_value());
    auto mixed_result = runner.buildInline(mixed);
    EXPECT_TRUE(mixed_result.spec_vocab_mask_cpu_owner.defined());
    mixed_result.ready_event->block(cuda_graph::graphGetCurrentStream());
    mixed_result.spec_cap_gpu.cpu();
    // Mutation after a descriptor acquisition changes only future eligibility.
    processor->think_infos_[0].pending_forced_think_end_token_ids = {3};
    processor->publishSpecSnapshotLocked();
    EXPECT_FALSE(processor->tryGetStaticThinkMaskDescriptor(current));
    EXPECT_EQ(retained->info.process_state, ThinkProcessState::NO_THINK);
}

TEST(StaticThinkSpecMaskTest, FreshOutputsSurviveFallbackAndOutstandingRounds) {
    SpecLogitsVerifyRunner runner;
    auto processor = stableThink(ThinkProcessState::NO_THINK);
    auto draft = torch::full({2, 4}, 10, torch::kInt64).cuda();
    std::vector<SpecLogitsVerifyRunner::LaunchResult> results;
    for (int round = 0; round < 6; ++round) {
        auto task = stableTask(draft, processor, 4, round % 2 ? 33 : 200064);
        results.push_back(runner.buildInline(task));
        if (round == 2) {
            task.active.push_back({std::make_shared<PatternProcessor>(2, 3), 1, 1, 5678});
            task.draft_tokens = torch::arange(10, 14, torch::kInt64).repeat({2, 1}).cuda();
            auto fallback = runner.buildInline(task);
            EXPECT_TRUE(fallback.spec_vocab_mask_cpu_owner.defined());
        }
    }
    for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];
        result.ready_event->block(cuda_graph::graphGetCurrentStream());
        EXPECT_TRUE(torch::equal(result.spec_cap_gpu.cpu(), torch::full({2}, 4, torch::kInt32)));
        for (size_t j = 0; j < i; ++j) {
            EXPECT_NE(result.spec_vocab_bitmask_gpu.data_ptr(), results[j].spec_vocab_bitmask_gpu.data_ptr());
            EXPECT_NE(result.spec_cap_gpu.data_ptr(), results[j].spec_cap_gpu.data_ptr());
        }
        EXPECT_TRUE(isMasked(result.spec_vocab_bitmask_gpu.cpu(), 0, 2, i % 2 ? 33 : 200064));
        result.consumed_event->record(cuda_graph::graphGetCurrentStream());
    }
}

// No CUDA calls in the callback. A watchdog prevents an unexpected host-side
// synchronization from hanging the suite; timeout is a test failure, never success.
struct StaticThinkProducerGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::atomic<bool> entered{false}, exited{false}, timed_out{false};
    void open() {
        { std::lock_guard<std::mutex> lock(mutex); release = true; }
        cv.notify_all();
    }
};

// Noncontiguous reshape may allocate/copy: it must follow the producer fence.
// The callback holds only its own stream; no production hooks or device-wide waits.
void checkDelayedFallbackDraft(int source_device) {
    c10::cuda::CUDAGuard runner_device(0);
    SpecLogitsVerifyRunner runner;
    runner.ensureBuffersFit(2, 4, 33, 2);
    auto base = torch::full({2, 4}, 2, torch::TensorOptions().dtype(torch::kInt64).device(torch::Device(torch::kCUDA, source_device)));
    auto draft = base.transpose(0, 1);
    ASSERT_FALSE(draft.is_contiguous());
    auto default_stream = at::cuda::getDefaultCUDAStream(source_device);
    auto producer = at::cuda::getStreamFromPool(false, source_device);
    default_stream.synchronize();
    // Warm the exact materialization allocation before the stream is held.
    { cuda_graph::GraphStreamGuard guard(default_stream); draft.reshape({2, 4}).contiguous(); }
    default_stream.synchronize();
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams=2; task.propose_step=4; task.vocab_size=33; task.draft_tokens=draft;
    task.draft_tokens_ready_event=std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    auto gate=std::make_shared<StaticThinkProducerGate>();
    // Declare future before cleanup: failure opens the producer before joining.
    std::future<void> work;
    struct ReleaseProducer {
        std::shared_ptr<StaticThinkProducerGate> gate;
        cuda_graph::GraphStream stream;
        ~ReleaseProducer() { gate->open(); try {stream.synchronize();} catch (...) {} }
    } cleanup{gate,producer};
    {
        cuda_graph::GraphStreamGuard guard(producer);
        auto* context=new std::shared_ptr<StaticThinkProducerGate>(gate);
        const auto queued=cudaLaunchHostFunc(producer.stream(),[](void* data) {
            std::unique_ptr<std::shared_ptr<StaticThinkProducerGate>> owner(static_cast<std::shared_ptr<StaticThinkProducerGate>*>(data));
            auto& gate=**owner;
            std::unique_lock<std::mutex> lock(gate.mutex);
            gate.entered.store(true,std::memory_order_release);gate.cv.notify_all();
            if(!gate.cv.wait_for(lock,std::chrono::seconds(10),[&]{return gate.release;}))gate.timed_out.store(true,std::memory_order_release);
            gate.exited.store(true,std::memory_order_release);
        },context);
        if(queued!=cudaSuccess)delete context;
        ASSERT_EQ(queued,cudaSuccess);
        base.fill_(10);
        task.draft_tokens_ready_event->record(producer);
    }
    {
        std::unique_lock<std::mutex> lock(gate->mutex);
        ASSERT_TRUE(gate->cv.wait_for(lock,std::chrono::seconds(2),[&]{return gate->entered.load(std::memory_order_acquire);}));
    }
    work=std::async(std::launch::async,[&] {
        c10::cuda::CUDAGuard device(0);
        runner.materializeDraftTokensToCpu(task);
    });
    // With the old same-device path the copy stream reaches its producer wait
    // after reshape on the default stream. With a synchronous foreign-device
    // fallback it may not reach that stream; the bound then releases producer.
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(100);
    while(runner.copy_stream_.query() && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    default_stream.synchronize();
    ASSERT_FALSE(gate->exited.load(std::memory_order_acquire));
    gate->open();
    ASSERT_EQ(work.wait_for(std::chrono::seconds(5)),std::future_status::ready);
    work.get();
    EXPECT_FALSE(gate->timed_out.load(std::memory_order_acquire));
    auto result=runner.draft_tokens_cpu_.reshape({-1}).narrow(0,0,8).reshape({2,4});
    EXPECT_TRUE(torch::equal(result,torch::full({2,4},10,torch::kInt32)));
}

TEST(SpecLogitsVerifyRunnerFallbackTest, NoncontiguousDraftReshapeWaitsForProducer) {
    checkDelayedFallbackDraft(0);
}

TEST(SpecLogitsVerifyRunnerFallbackTest, ForeignDeviceDraftWaitsForProducer) {
    ASSERT_GE(at::cuda::device_count(),2);
    checkDelayedFallbackDraft(1);
}

TEST(StaticThinkSpecMaskTest, DelayedProducerSurvivesSourceAndDescriptorOwnerRelease) {
    SpecLogitsVerifyRunner runner;
    auto processor = stableThink(ThinkProcessState::AFTER_THINK);
    auto gpu_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto pinned_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    // Warm event creation, descriptor uploads, mask/cap and both allocators before
    // the callback blocks. Reserve cached pressure blocks before releasing them.
    auto draft = torch::full({2, 4}, 10, gpu_options);
    draft.index_put_({0, 1}, 3);
    auto task = stableTask(draft, processor, 4);
    for (int i = 0; i < 3; ++i) {
        auto warm = runner.buildInline(task);
        warm.ready_event->block(cuda_graph::graphGetCurrentStream());
        warm.spec_cap_gpu.cpu();
    }
    std::vector<torch::Tensor> gpu_pressure, host_pressure;
    gpu_pressure.reserve(32);
    host_pressure.reserve(32);
    for (int i = 0; i < 32; ++i) {
        gpu_pressure.push_back(torch::empty({2, 4}, gpu_options));
        host_pressure.push_back(torch::empty({2, 3}, pinned_options));
    }
    cuda_graph::graphGetCurrentStream().synchronize();
    runner.copy_stream_.synchronize();
    gpu_pressure.clear();
    host_pressure.clear();
    auto gate = std::make_shared<StaticThinkProducerGate>();
    // This guard covers every ASSERT/exception after enqueue, including failure
    // to observe callback entry. It releases first, then joins the copy stream.
    struct ReleaseAndJoin {
        std::shared_ptr<StaticThinkProducerGate> gate;
        torch::Stream stream;
        ~ReleaseAndJoin() {
            gate->open();
            try { stream.synchronize(); } catch (...) {}
        }
    } cleanup{gate, runner.copy_stream_};
    auto* context = new std::shared_ptr<StaticThinkProducerGate>(gate);
    const auto queued = cudaLaunchHostFunc(cuda_graph::toGraphStream(runner.copy_stream_).stream(), [](void* ptr) {
        std::unique_ptr<std::shared_ptr<StaticThinkProducerGate>> owner(
            static_cast<std::shared_ptr<StaticThinkProducerGate>*>(ptr));
        auto& gate = **owner;
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.entered.store(true, std::memory_order_release);
        gate.cv.notify_all();
        if (!gate.cv.wait_for(lock, std::chrono::seconds(10), [&] { return gate.release; }))
            gate.timed_out.store(true, std::memory_order_release);
        gate.exited.store(true, std::memory_order_release);
    }, context);
    if (queued != cudaSuccess) delete context;
    ASSERT_EQ(queued, cudaSuccess);
    {
        std::unique_lock<std::mutex> lock(gate->mutex);
        ASSERT_TRUE(gate->cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return gate->entered.load(std::memory_order_acquire);
        }));
    }
    auto result = runner.buildInline(task);
    ASSERT_TRUE(result.static_think_descriptor_cpu_owner.defined());
    ASSERT_FALSE(gate->exited.load(std::memory_order_acquire));
    auto mask = result.spec_vocab_bitmask_gpu;
    auto cap = result.spec_cap_gpu;
    auto ready = result.ready_event;
    // The gate holds the stream BEFORE all descriptor/draft reads. Neither a
    // long host allocation nor a sleep duration can turn this into a completed-
    // producer owner-release test without failing the pending assertions.
    result = SpecLogitsVerifyRunner::LaunchResult();
    task = SpecLogitsVerifyRunner::LaunchTask();
    draft = torch::Tensor();
    ASSERT_FALSE(gate->exited.load(std::memory_order_acquire));
    for (int i = 0; i < 32; ++i) {
        gpu_pressure.push_back(torch::full({2, 4}, 63, gpu_options));
        host_pressure.push_back(torch::full({2, 3}, 63, pinned_options));
    }
    cuda_graph::graphGetCurrentStream().synchronize();
    ASSERT_FALSE(gate->exited.load(std::memory_order_acquire));
    gate->open();
    ready->block(cuda_graph::graphGetCurrentStream());
    auto cap_cpu = cap.cpu();
    EXPECT_EQ(cap_cpu.data_ptr<int32_t>()[0], 1);
    EXPECT_EQ(cap_cpu.data_ptr<int32_t>()[1], 4);
    auto mask_cpu = mask.cpu();
    EXPECT_TRUE(isMasked(mask_cpu, 1, 3, 33));
    EXPECT_FALSE(isMasked(mask_cpu, 2, 3, 33));
    EXPECT_TRUE(gate->exited.load(std::memory_order_acquire));
    EXPECT_FALSE(gate->timed_out.load(std::memory_order_acquire));
}


#endif

INSTANTIATE_TEST_SUITE_P(ShortVerifyBudgets, SpecLogitsVerifyRunnerTest, ::testing::Values(1, 4, 7));

}  // namespace
}  // namespace rtp_llm
