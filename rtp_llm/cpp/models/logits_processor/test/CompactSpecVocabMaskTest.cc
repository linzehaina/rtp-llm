#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/compact_spec_vocab_mask.h"

#include <gtest/gtest.h>
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <vector>

namespace rtp_llm {
namespace {

struct Masks {
    torch::Tensor dense;
    torch::Tensor compact;
};

Masks makeMasks(int64_t rows, int64_t vocab) {
    const int64_t words = (vocab + 31) / 32;
    auto bits = torch::zeros({rows, words}, torch::TensorOptions().dtype(torch::kInt32));
    auto dense = torch::empty({rows, vocab}, torch::TensorOptions().dtype(torch::kBool));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t token = 0; token < vocab; ++token) {
            // All-denied, all-allowed, single-token-allowed, and mixed rows.
            const bool allow = row % 4 == 1 || (row % 4 == 2 && token == vocab - 1)
                               || (row % 4 == 3 && (token * 17 + row) % 7 < 3);
            dense.data_ptr<bool>()[row * vocab + token] = !allow;
            if (allow) {
                const auto offset = row * words + token / 32;
                bits.data_ptr<int32_t>()[offset] = static_cast<int32_t>(
                    static_cast<uint32_t>(bits.data_ptr<int32_t>()[offset]) | (uint32_t{1} << (token % 32)));
            }
        }
        // Garbage allowed tail bits must never affect the neighboring row.
        if (vocab % 32) {
            const auto offset = (row + 1) * words - 1;
            bits.data_ptr<int32_t>()[offset] = static_cast<int32_t>(
                static_cast<uint32_t>(bits.data_ptr<int32_t>()[offset]) | (~uint32_t{0} << (vocab % 32)));
        }
    }
    return {dense.to(torch::kCUDA), bits.to(torch::kCUDA)};
}

void expectSame(const torch::Tensor& actual, const torch::Tensor& expected) {
    // Preserve allowed NaNs and infinities; compare every other value exactly.
    if (actual.scalar_type() == torch::kFloat32) {
        EXPECT_TRUE(torch::equal(actual.view(torch::kInt32), expected.view(torch::kInt32)));
    }
    EXPECT_TRUE(torch::equal(torch::isnan(actual), torch::isnan(expected)));
    auto valid = torch::logical_not(torch::isnan(expected));
    EXPECT_TRUE(torch::equal(actual.masked_select(valid), expected.masked_select(valid)));
}

TEST(CompactSpecVocabMaskTest, DenseEquivalentAcrossDtypesShapesAndStrides) {
    LogitsProcessorStates processors;
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16, torch::kFloat64}) {
        for (int64_t vocab : {1, 4, 31, 32, 33, 1028, 200064}) {
            const int64_t rows = vocab == 200064 ? 160 : 5;
            auto masks = makeMasks(rows, vocab);
            for (int layout : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " vocab=" << vocab
                                                  << " layout=" << layout);
                auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
                auto logits = layout == 0 ? torch::randn({rows, vocab}, options)
                              : layout == 1 ? torch::randn({vocab, rows}, options).transpose(0, 1)
                                            : torch::randn({rows, vocab * 2}, options).slice(1, 0, vocab * 2, 2);
                if (vocab > 1) {
                    if (dtype == torch::kFloat32) {
                        logits.view(torch::kInt32).index_put_({1, 0}, int32_t{0x7fc12345});
                    } else {
                        logits.index_put_({1, 0}, std::numeric_limits<float>::quiet_NaN());
                    }
                    logits.index_put_({1, 1}, std::numeric_limits<float>::infinity());
                    if (dtype == torch::kFloat32 && vocab > 2) {
                        // Row 3/token 2 is allowed inside a mixed allow nibble,
                        // so the vector load/store must preserve its NaN bits.
                        logits.view(torch::kInt32).index_put_({3, 2}, int32_t{0x7fc12345});
                    }
                }
                torch::Tensor expected;
                try {
                    expected = logits.clone().masked_fill_(masks.dense, BaseLogitsProcessor::neg_inf);
                } catch (const std::exception&) {
                    // Preserve legacy overflow rejection rather than silently
                    // supporting a dtype by substituting an infinity sentinel.
                    EXPECT_ANY_THROW(applyCompactSpecVocabMask(logits, masks.compact));
                    continue;
                }
                SamplerInputs inputs;
                inputs.logits = logits;
                inputs.phase = LogitsProcessorPhase::MTP_VERIFY;
                inputs.spec_vocab_bitmask_gpu = masks.compact;
                processors.batchProcess(inputs);
                expectSame(logits, expected);

                auto legacy = expected.clone();
                inputs.logits = legacy;
                inputs.spec_vocab_bitmask_gpu = torch::Tensor();
                inputs.spec_vocab_mask_gpu = masks.dense;
                processors.batchProcess(inputs);
                expectSame(legacy, expected);
            }
        }
    }
}

TEST(CompactSpecVocabMaskTest, Float4AlignmentAndLargeRowFallback) {
    // Include aligned row padding, an unaligned base, an unaligned row pitch,
    // and a batch beyond CUDA grid.y. Every case must preserve exact FP32 bits.
    for (int layout : {0, 1, 2, 3}) {
        const int64_t rows = layout == 3 ? 65536 : 5;
        constexpr int64_t vocab = 36;
        auto masks = makeMasks(rows, vocab);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        const int64_t pitch = layout == 1 ? vocab + 1 : vocab + 4;
        auto storage = torch::randn({rows, pitch}, options);
        const int64_t offset = layout == 2 ? 1 : 0;
        auto logits = storage.slice(1, offset, offset + vocab);
        logits.view(torch::kInt32).index_put_({1, 0}, int32_t{0x7fc12345});
        auto expected = logits.clone().masked_fill_(masks.dense, BaseLogitsProcessor::neg_inf);
        applyCompactSpecVocabMask(logits, masks.compact);
        expectSame(logits, expected);
    }
}

TEST(CompactSpecVocabMaskTest, AllDeniedPreservesLegacyFiniteSentinelAndDistribution) {
    auto masks = makeMasks(1, 33);
    auto logits = torch::randn({1, 33}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto expected = logits.clone().masked_fill_(masks.dense, BaseLogitsProcessor::neg_inf);
    applyCompactSpecVocabMask(logits, masks.compact);
    EXPECT_TRUE(torch::equal(logits, expected));
    EXPECT_TRUE(torch::isfinite(logits).all().item<bool>());
    EXPECT_TRUE(torch::equal(logits.softmax(-1), expected.softmax(-1)));
    EXPECT_TRUE(torch::isfinite(logits.softmax(-1)).all().item<bool>());
}

TEST(CompactSpecVocabMaskTest, OrdinaryPhaseIgnoresMaskAndRejectsAmbiguousVerifyInput) {
    LogitsProcessorStates processors;
    auto masks = makeMasks(5, 33);
    SamplerInputs inputs;
    inputs.logits = torch::ones({5, 33}, torch::TensorOptions().device(torch::kCUDA));
    inputs.spec_vocab_bitmask_gpu = masks.compact;
    auto original = inputs.logits.clone();
    processors.batchProcess(inputs);
    EXPECT_TRUE(torch::equal(inputs.logits, original));
    inputs.phase = LogitsProcessorPhase::MTP_VERIFY;
    inputs.spec_vocab_mask_gpu = masks.dense;
    EXPECT_THROW(processors.batchProcess(inputs), c10::Error);
}

TEST(CompactSpecVocabMaskTest, RejectsInvalidCompactContractAndHandlesEmpty) {
    auto logits = torch::ones({5, 33}, torch::TensorOptions().device(torch::kCUDA));
    auto masks = makeMasks(5, 33);
    EXPECT_THROW(applyCompactSpecVocabMask(logits, masks.dense), c10::Error);
    EXPECT_THROW(applyCompactSpecVocabMask(logits, masks.compact.narrow(0, 0, 4)), c10::Error);
    EXPECT_THROW(applyCompactSpecVocabMask(logits, masks.compact.to(torch::kCPU)), c10::Error);
    EXPECT_THROW(applyCompactSpecVocabMask(logits, masks.compact.transpose(0, 1)), c10::Error);
    auto alias = torch::ones({1, 33}, logits.options()).expand({5, 33});
    EXPECT_THROW(applyCompactSpecVocabMask(alias, masks.compact), c10::Error);
    applyCompactSpecVocabMask(torch::empty({0, 33}, logits.options()), masks.compact.narrow(0, 0, 0));
}

TEST(CompactSpecVocabMaskTest, StaticThinkProducerEagerAndGraphReplayReadsFreshDraft) {
    auto stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard guard(stream);
    for (auto dtype : {torch::kInt32, torch::kInt64}) {
        auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
        auto draft = torch::full({3, 4}, 7, options.dtype(dtype));
        auto descriptors = torch::tensor({{1, 2, 3}, {1, 63, 3}, {0, 2, 3}}, torch::kInt32).cuda();
        auto mask = torch::empty({15, 2}, options);
        auto cap = torch::empty({3}, options);
        buildStaticThinkSpecMask(draft, descriptors, mask, cap);
        stream.synchronize();
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        buildStaticThinkSpecMask(draft, descriptors, mask, cap);
        graph.capture_end();
        for (int blocked = -1; blocked < 4; ++blocked) {
            draft.fill_(7);
            if (blocked >= 0) {
                draft.index_put_({0, blocked}, 2);
                draft.index_put_({1, blocked}, 63);
                draft.index_put_({2, blocked}, -1); // Inactive stream ignores draft invalidity.
            }
            graph.replay();
            auto cap_cpu = cap.cpu();
            auto mask_cpu = mask.cpu();
            const int expected = blocked < 0 ? 4 : blocked;
            EXPECT_EQ(cap_cpu.data_ptr<int32_t>()[0], expected);
            EXPECT_EQ(cap_cpu.data_ptr<int32_t>()[1], expected);
            EXPECT_EQ(cap_cpu.data_ptr<int32_t>()[2], 4);
            for (int b = 0; b < 3; ++b) {
                for (int row = 0; row <= 4; ++row) {
                    for (int word = 0; word < 2; ++word) {
                        uint32_t expected_word = ~uint32_t{0};
                        if (b < 2 && row <= expected) {
                            for (int token : {b == 0 ? 2 : 63, 3}) {
                                if (token / 32 == word) expected_word &= ~(uint32_t{1} << (token % 32));
                            }
                        }
                        EXPECT_EQ(static_cast<uint32_t>(mask_cpu.data_ptr<int32_t>()[(b * 5 + row) * 2 + word]), expected_word);
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace rtp_llm
