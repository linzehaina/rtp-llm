#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
using namespace std;
using namespace rtp_llm;

namespace {

torch::Tensor normalizedRandomProbs(int64_t batch_size, int64_t vocab_size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto probs   = torch::rand({batch_size, vocab_size}, options);
    return probs.div(probs.sum(-1, /*keepdim=*/true));
}

void assertAllValid(const torch::Tensor& valid) {
    ASSERT_TRUE(valid.defined());
    ASSERT_EQ(valid.scalar_type(), torch::kBool);
    ASSERT_TRUE(valid.cpu().all().item<bool>());
}

void assertTokenIn(const std::vector<int32_t>& token_ids, size_t index, std::initializer_list<int32_t> allowed) {
    ASSERT_NE(std::find(allowed.begin(), allowed.end(), token_ids[index]), allowed.end())
        << "token_ids[" << index << "]=" << token_ids[index];
}

void assertCumLogProbMatches(const std::vector<float>&   cum_log_probs,
                             const std::vector<float>&   output_all_probs,
                             const std::vector<int32_t>& token_ids,
                             const std::vector<float>&   initial_cum_log_probs,
                             int64_t                     batch_size,
                             int64_t                     step,
                             int64_t                     vocab_size) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const auto token = token_ids[batch * (step + 1) + step];
        ASSERT_GE(token, 0);
        ASSERT_LT(token, vocab_size);
        const auto prob = output_all_probs[batch * vocab_size + token];
        ASSERT_GT(prob, 0.0f);
        ASSERT_NEAR(cum_log_probs[batch], initial_cum_log_probs[batch] + std::log(prob), 1e-3);
    }
}

}  // namespace

class CudaSamplerTest: public DeviceTestBase {
public:
protected:
    void assertBatchTemperatureBeyondBlock(torch::ScalarType dtype) {
        constexpr int64_t vocab = 32;  // The kernel launches only 32 threads per block.
        for (const int64_t batch : {80, 128}) {
            SCOPED_TRACE(::testing::Message() << "batch=" << batch << " dtype=" << dtype);
            std::vector<float> logits_data(batch * vocab), temperature_data(batch), inverse_data(batch);
            for (int64_t row = 0; row < batch; ++row) {
                temperature_data[row] = 0.5f + static_cast<float>(row % 17) / 10.0f;
                inverse_data[row]     = 1.0f / (temperature_data[row] + 1e-6f);
                for (int64_t col = 0; col < vocab; ++col) {
                    logits_data[row * vocab + col] = static_cast<float>(1 + (row * 7 + col * 3) % 53) / 8.0f;
                }
            }
            auto input_cpu = torch::tensor(logits_data, torch::kFloat32).reshape({batch, vocab}).to(dtype);
            // The even-vocabulary FP16 path rounds the reciprocal to half before
            // multiplying in half2. Match both roundings, not float division.
            auto       inverse_cpu  = torch::tensor(inverse_data, torch::kFloat32).to(dtype).to(torch::kFloat32);
            auto       expected     = (input_cpu.to(torch::kFloat32) * inverse_cpu.unsqueeze(1)).to(dtype);
            auto       logits       = input_cpu.cuda();
            auto       temperatures = cudaTensor(temperature_data, {batch});
            const auto stream       = at::cuda::getCurrentCUDAStream().stream();
            if (dtype == torch::kFloat32) {
                invokeBatchApplyTemperaturePenalty<float>(
                    logits.data_ptr<float>(), nullptr, temperatures.data_ptr<float>(), batch, vocab, vocab, stream);
            } else {
                invokeBatchApplyTemperaturePenalty<half>(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                                                         nullptr,
                                                         temperatures.data_ptr<float>(),
                                                         batch,
                                                         vocab,
                                                         vocab,
                                                         stream);
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            auto actual = logits.cpu();
            ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
            if (dtype == torch::kFloat16) {
                ASSERT_TRUE(torch::equal(actual, expected));
            } else {
                ASSERT_TRUE(torch::allclose(actual, expected, /*rtol=*/1e-6, /*atol=*/1e-6));
            }
        }
    }

    // This is a lowest-layer equivalence check, not permission to omit history
    // for penalties, processors, beam search, or arbitrary Sampler consumers.
    void assertCompactHistoryEquivalent(int64_t batch,
                                        int64_t history,
                                        int     mode,
                                        bool    with_all_probs      = true,
                                        int64_t vocab               = 32,
                                        bool    exact_probabilities = true) {
        SCOPED_TRACE(::testing::Message() << "batch=" << batch << " history=" << history << " mode=" << mode
                                          << " all_probs=" << with_all_probs << " vocab=" << vocab);
        ASSERT_TRUE(with_all_probs || mode == 0);
        std::vector<float>   logits_data(batch * vocab);
        std::vector<int32_t> input_data(batch), sequence_data(batch), top_k_data(batch);
        std::vector<float>   top_p_data(batch), temperature_data(batch);
        for (int64_t row = 0; row < batch; ++row) {
            input_data[row]       = static_cast<int32_t>(history - 4 - row % 3);
            sequence_data[row]    = static_cast<int32_t>(history - row % 3);
            top_k_data[row]       = mode == 0 ? 1 : (row % 2 ? 8 : 0);
            top_p_data[row]       = mode == 0 ? 1.0f : (row % 2 ? 0.85f : 0.7f);
            temperature_data[row] = mode == 0 ? 1.0f : (row % 2 ? 0.8f : 1.2f);
            if (mode == 2 && row % 2 == 0) {
                // Match fillSamplerCommonInputs' do_sample=false normalization.
                top_k_data[row]       = 1;
                top_p_data[row]       = 1.0f;
                temperature_data[row] = 1.0f;
            }
            for (int64_t col = 0; col < vocab; ++col) {
                logits_data[row * vocab + col] = static_cast<float>((col * 7 + row * 3) % 31) / 9.0f;
            }
        }
        auto logits           = cudaTensor(logits_data, {batch, vocab});
        auto input_lengths    = torch::tensor(input_data, torch::kInt32);
        auto sequence_lengths = torch::tensor(sequence_data, torch::kInt32);
        auto input_before     = input_lengths.clone();
        auto sequence_before  = sequence_lengths.clone();
        auto do_sample        = torch::ones({batch}, torch::kBool).pin_memory();
        for (int64_t row = 0; row < batch; ++row) {
            do_sample.data_ptr<bool>()[row] = mode != 2 || row % 2 != 0;
        }
        std::vector<at::Generator> full_generators, compact_generators;
        for (int64_t row = 0; row < batch; ++row) {
            full_generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            compact_generators.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
            full_generators.back().set_current_seed(9100 + row);
            compact_generators.back().set_current_seed(9100 + row);
        }
        // Two successive calls verify continuation, not just an equal first draw.
        for (int round = 0; round < 2; ++round) {
            SCOPED_TRACE(::testing::Message() << "round=" << round);
            auto device_i32     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto full_tokens    = torch::full({batch, history + 1}, 7, device_i32);
            auto compact_tokens = torch::full({batch, 1}, -1, device_i32);
            full_tokens.select(1, history).fill_(-1);
            auto                       full_logits    = logits.clone();
            auto                       compact_logits = logits.clone();
            auto                       full_probs     = torch::zeros_like(logits);
            auto                       compact_probs  = torch::zeros_like(logits);
            auto                       full_cum       = torch::full({batch}, -0.5f, logits.options());
            auto                       compact_cum    = full_cum.clone();
            std::vector<torch::Tensor> before_rng;
            for (int64_t row = 0; row < batch; ++row) {
                before_rng.push_back(full_generators[row].get_state().clone());
            }

            auto make_params = [&](torch::Tensor                     l,
                                   torch::Tensor                     tokens,
                                   torch::Tensor                     probs,
                                   torch::Tensor                     cum,
                                   const std::vector<at::Generator>& generators) {
                return GreedyParams{l,
                                    input_lengths,
                                    sequence_lengths,
                                    tokens,
                                    static_cast<size_t>(history),
                                    pinnedIntTensor(top_k_data),
                                    pinnedFloatTensor(top_p_data),
                                    pinnedFloatTensor(temperature_data),
                                    pinnedFloatTensor(std::vector<float>(batch, 1.0f)),
                                    pinnedIntTensor(std::vector<int32_t>(batch, 0)),
                                    cum,
                                    std::nullopt,
                                    false,
                                    with_all_probs ? std::optional<torch::Tensor>(probs) : std::nullopt,
                                    pinnedFloatTensor(std::vector<float>(batch, 0.0f)),
                                    pinnedFloatTensor(std::vector<float>(batch, 0.0f)),
                                    do_sample,
                                    generators};
            };
            auto full_params = make_params(full_logits, full_tokens, full_probs, full_cum, full_generators);
            auto compact_params =
                make_params(compact_logits, compact_tokens, compact_probs, compact_cum, compact_generators);
            ASSERT_EQ(full_params.output_all_probs.has_value(), with_all_probs);
            ASSERT_EQ(compact_params.output_all_probs.has_value(), with_all_probs);
            // Logical step and lengths intentionally remain real, including 81920.
            ASSERT_EQ(full_params.step, compact_params.step);
            auto full_output    = execSampleGreedy(full_params);
            auto compact_output = execSampleGreedy(compact_params);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_TRUE(torch::equal(full_tokens.select(1, history), compact_tokens.select(1, 0)));
            ASSERT_TRUE(compact_tokens.ge(0).logical_and(compact_tokens.lt(vocab)).all().item<bool>());
            if (exact_probabilities) {
                ASSERT_TRUE(torch::equal(full_probs, compact_probs));
                ASSERT_TRUE(torch::equal(full_cum, compact_cum));
            } else {
                // Real-vocab multi-CTA renorm is not bitwise repeatable even
                // full/full with frozen input. Stage-isolation tests retain
                // that evidence. Bound distribution drift per row, not only
                // per-element absolute error (which hides broad mass shifts).
                ASSERT_TRUE(torch::isfinite(full_probs).all().item<bool>());
                ASSERT_TRUE(torch::isfinite(compact_probs).all().item<bool>());
                ASSERT_TRUE(torch::equal(full_probs.gt(0), compact_probs.gt(0)));
                ASSERT_LE((full_probs - compact_probs).abs().sum(-1).max().item<float>(), 1e-6f);
                ASSERT_LE((full_probs.sum(-1) - 1).abs().max().item<float>(), 1e-6f);
                ASSERT_LE((compact_probs.sum(-1) - 1).abs().max().item<float>(), 1e-6f);
                ASSERT_LE((full_cum - compact_cum).abs().max().item<float>(), 2e-6f);
            }
            ASSERT_TRUE(torch::equal(full_logits, compact_logits));
            ASSERT_EQ(full_output.success.defined(), compact_output.success.defined());
            if (full_output.success.defined()) {
                ASSERT_TRUE(torch::equal(full_output.success, compact_output.success));
                ASSERT_TRUE(full_output.success.all().item<bool>());
            }
            if (!with_all_probs) {
                // top1's no-all_probs fast path is argmax only: no softmax or RNG draw.
                ASSERT_FALSE(full_output.success.defined());
                ASSERT_FALSE(compact_output.success.defined());
                ASSERT_TRUE(torch::equal(compact_tokens.select(1, 0), logits.argmax(-1).to(torch::kInt32)));
                ASSERT_TRUE(torch::equal(full_logits, logits));
                ASSERT_TRUE(full_cum.eq(-0.5f).all().item<bool>());
            }
            ASSERT_TRUE(full_tokens.narrow(1, 0, history).eq(7).all().item<bool>());
            ASSERT_TRUE(torch::equal(input_lengths, input_before));
            ASSERT_TRUE(torch::equal(sequence_lengths, sequence_before));
            for (int64_t row = 0; row < batch; ++row) {
                ASSERT_TRUE(torch::equal(full_generators[row].get_state(), compact_generators[row].get_state()));
                if (!with_all_probs) {
                    ASSERT_TRUE(torch::equal(before_rng[row], full_generators[row].get_state()));
                    ASSERT_TRUE(torch::equal(before_rng[row], compact_generators[row].get_state()));
                }
            }
            // Even top1 with all_probs must retain the current RNG reservation.
            if (with_all_probs) {
                ASSERT_FALSE(torch::equal(before_rng.front(), full_generators.front().get_state()));
            }
        }
    }

    // Helper: create a CUDA tensor from float data
    torch::Tensor cudaTensor(std::vector<float> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kFloat32).reshape(shape).to(torch::kCUDA);
    }

    // Helper: create a CUDA tensor from int32 data
    torch::Tensor cudaIntTensor(std::vector<int32_t> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kInt32).reshape(shape).to(torch::kCUDA);
    }

    // Helper: create a pinned CPU tensor from int32 data (for HOST buffers)
    torch::Tensor pinnedIntTensor(std::vector<int32_t> data) {
        return torch::tensor(data, torch::kInt32).pin_memory();
    }

    // Helper: create a pinned CPU tensor from float data
    torch::Tensor pinnedFloatTensor(std::vector<float> data) {
        return torch::tensor(data, torch::kFloat32).pin_memory();
    }

    // Helper: read GPU int32 tensor to host vector
    std::vector<int32_t> toHostInt(const torch::Tensor& t) {
        auto cpu = t.cpu().contiguous();
        return std::vector<int32_t>(cpu.data_ptr<int32_t>(), cpu.data_ptr<int32_t>() + cpu.numel());
    }

    // Helper: read GPU float tensor to host vector
    std::vector<float> toHostFloat(const torch::Tensor& t) {
        auto cpu = t.cpu().contiguous();
        return std::vector<float>(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
    }
};

TEST_F(CudaSamplerTest, CompactVerifyRowsPreserveLiveSamplesAndDenseRng) {
    constexpr int64_t width = 8;
    for (int64_t vocab : {256, 200064}) {
        for (int64_t requests : {1, 3, 4}) {
            const auto dense_rows            = requests * width;
            const bool use_default_generator = requests == 3;
            for (int mode = 0; mode < 4; ++mode) {
                SCOPED_TRACE(::testing::Message() << "vocab=" << vocab << " requests=" << requests << " mode=" << mode);
                std::vector<at::Generator> full_generators, compact_generators;
                for (int64_t request = 0; request < (use_default_generator ? 0 : requests); ++request) {
                    auto full_gen    = torch::make_generator<at::CUDAGeneratorImpl>();
                    auto compact_gen = torch::make_generator<at::CUDAGeneratorImpl>();
                    full_gen.set_current_seed(2300 + request);
                    compact_gen.set_current_seed(2300 + request);
                    for (int64_t position = 0; position < width; ++position) {
                        full_generators.push_back(full_gen);
                        compact_generators.push_back(compact_gen);
                    }
                }
                // Changing ragged prefixes across calls exercises RNG continuation
                // and full-width / one-row boundaries without resetting generators.
                for (int round = 0; round < 3; ++round) {
                    std::vector<int64_t> live;
                    for (int64_t request = 0; request < requests; ++request) {
                        const auto length = round == 0 ? 1 : round == 1 ? width : 2 + request;
                        for (int64_t position = 0; position < length; ++position)
                            live.push_back(request * width + position);
                    }
                    auto                 rows       = torch::tensor(live, torch::kLong).to(torch::kCUDA);
                    auto                 device_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
                    auto                 logits = torch::randn({dense_rows, vocab}, device_i32.dtype(torch::kFloat32));
                    auto                 full_tokens    = torch::zeros({dense_rows, 1}, device_i32);
                    auto                 compact_tokens = full_tokens.clone();
                    auto                 full_probs     = torch::empty_like(logits);
                    auto                 compact_probs  = torch::empty_like(logits);
                    std::vector<int32_t> top_k(dense_rows, mode == 0 ? 1 : mode == 2 ? 5 : 0);
                    std::vector<float>   top_p(dense_rows, mode == 1 ? 0.8f : 1.0f);
                    if (mode == 3) {
                        for (int64_t row = 0; row < dense_rows; ++row) {
                            top_k[row] = row % width < 2 ? 0 : 5;
                            top_p[row] = row % width < 2 ? 0.8f : 0.9f;
                        }
                    }
                    auto make_params =
                        [&](torch::Tensor tokens, torch::Tensor probs, const std::vector<at::Generator>& generators) {
                            return GreedyParams{logits.clone(),
                                                pinnedIntTensor(std::vector<int32_t>(dense_rows, 81920)),
                                                pinnedIntTensor(std::vector<int32_t>(dense_rows, 81920)),
                                                tokens,
                                                81920,
                                                pinnedIntTensor(top_k),
                                                pinnedFloatTensor(top_p),
                                                pinnedFloatTensor(std::vector<float>(dense_rows, 0.7f)),
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                false,
                                                probs,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                generators};
                        };
                    auto full                     = make_params(full_tokens, full_probs, full_generators);
                    auto compact                  = make_params(compact_tokens, compact_probs, compact_generators);
                    compact.sample_rows           = rows;
                    auto       default_generator  = at::cuda::detail::getDefaultCUDAGenerator();
                    const auto default_before     = default_generator.get_state().clone();
                    auto       full_output        = execSampleGreedy(full);
                    const auto default_after_full = default_generator.get_state().clone();
                    if (use_default_generator)
                        default_generator.set_state(default_before);
                    auto compact_output = execSampleGreedy(compact);
                    EXPECT_TRUE(torch::equal(default_after_full, default_generator.get_state()));
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    EXPECT_TRUE(torch::equal(full_tokens.index_select(0, rows), compact_tokens.index_select(0, rows)));
                    const auto live_full    = full_probs.index_select(0, rows);
                    const auto live_compact = compact_probs.index_select(0, rows);
                    if (vocab == 256 || mode < 2) {
                        EXPECT_TRUE(torch::equal(live_full, live_compact));
                    } else {
                        // Large-vocab TopK stays on the dense executor path.
                        const auto delta = (live_full - live_compact).abs();
                        EXPECT_TRUE(torch::equal(live_full > 0, live_compact > 0));
                        EXPECT_LE(delta.sum(-1).max().item<float>(), 1e-6f);

                    }
                    EXPECT_TRUE(torch::equal(full.logits.index_select(0, rows), compact.logits.index_select(0, rows)));
                    EXPECT_TRUE(torch::allclose(compact_probs.sum(-1), torch::ones({dense_rows}, logits.options())));
                    ASSERT_EQ(full_output.success.defined(), compact_output.success.defined());
                    if (full_output.success.defined()) {
                        EXPECT_TRUE(torch::equal(full_output.success.index_select(0, rows),
                                                 compact_output.success.index_select(0, rows)));
                    }
                    for (int64_t request = 0; request < (use_default_generator ? 0 : requests); ++request) {
                        EXPECT_TRUE(torch::equal(full_generators[request * width].get_state(),
                                                 compact_generators[request * width].get_state()));
                    }
                }
            }
        }
    }
}

TEST_F(CudaSamplerTest, LargeVocabTopPRenormPreservesSupportAcrossRepeatsAndCompactRows) {
    constexpr int64_t vocab = 200064, width = 8;
    const auto        options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto        stream  = at::cuda::getCurrentCUDAStream().stream();
    for (int64_t requests : {3, 4, 16}) {
        const auto           dense_rows = requests * width;
        auto                 probs      = torch::softmax(torch::randn({dense_rows, vocab}, options), -1);
        std::vector<int64_t> live;
        for (int64_t request = 0; request < requests; ++request)
            for (int64_t position = 0; position < 1 + request % width; ++position)
                live.push_back(request * width + position);
        auto rows          = torch::tensor(live, torch::kLong).cuda();
        auto compact_probs = probs.index_select(0, rows);
        for (float p : {0.8f, 0.95f, 1.0f}) {
            SCOPED_TRACE(::testing::Message() << "requests=" << requests << " top_p=" << p);
            auto reference = torch::empty_like(probs);
            top_p_renorm_probs(probs, reference, std::nullopt, p, (int64_t)stream);
            for (int repeat = 0; repeat < 5; ++repeat) {
                auto actual  = torch::empty_like(probs);
                auto compact = torch::empty_like(compact_probs);
                top_p_renorm_probs(probs, actual, std::nullopt, p, (int64_t)stream);
                top_p_renorm_probs(compact_probs, compact, std::nullopt, p, (int64_t)stream);
                ASSERT_TRUE(torch::equal(reference, actual));
                ASSERT_TRUE(torch::equal(reference.index_select(0, rows), compact));
                ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
                ASSERT_TRUE(torch::allclose(actual.sum(-1), torch::ones({dense_rows}, options)));
            }
        }
    }
}

TEST_F(CudaSamplerTest, CompactVerifyRowsPreserveCappedRejectionPrefix) {
    constexpr int64_t    batch = 4, gamma = 7, width = 8, vocab = 256, dense_rows = batch * width;
    auto                 gpu_f32     = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto                 gpu_i32     = gpu_f32.dtype(torch::kInt32);
    auto                 lengths_cpu = torch::tensor({1, 3, 5, 8}, torch::kInt32);
    auto                 lengths     = lengths_cpu.cuda();
    std::vector<int64_t> live;
    for (int64_t request = 0; request < batch; ++request)
        for (int64_t position = 0; position < lengths_cpu.data_ptr<int32_t>()[request]; ++position)
            live.push_back(request * width + position);
    auto rows = torch::tensor(live, torch::kLong).cuda();
    for (bool do_sample : {false, true}) {
        for (bool full_agreement : {false, true}) {
            for (int trial = 0; trial < 8; ++trial) {
                auto frozen      = torch::randn({dense_rows, vocab}, gpu_f32);
                auto make_params = [&]() {
                    GreedyParams params{frozen.clone(),
                                        pinnedIntTensor(std::vector<int32_t>(dense_rows, 81920)),
                                        pinnedIntTensor(std::vector<int32_t>(dense_rows, 81920)),
                                        torch::zeros({dense_rows, 1}, gpu_i32),
                                        81920,
                                        pinnedIntTensor(std::vector<int32_t>(dense_rows, do_sample ? 0 : 1)),
                                        pinnedFloatTensor(std::vector<float>(dense_rows, 0.95f)),
                                        pinnedFloatTensor(std::vector<float>(dense_rows, 0.7f)),
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        false,
                                        torch::empty_like(frozen)};
                    for (int64_t row = 0; row < dense_rows; ++row) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(3200 + row + trial * dense_rows);
                        params.generator.push_back(generator);
                    }
                    return params;
                };
                auto full           = make_params();
                auto compact        = make_params();
                compact.sample_rows = rows;
                execSampleGreedy(full);
                execSampleGreedy(compact);
                auto                       target_probs = full.output_all_probs.value().reshape({batch, width, vocab});
                auto                       draft_probs  = full_agreement ?
                                                              target_probs.narrow(1, 0, gamma).contiguous() :
                                                              normalizedRandomProbs(batch * gamma, vocab).reshape({batch, gamma, vocab});
                auto                       draft_ids    = full_agreement ?
                                                              full.token_ids.reshape({batch, width}).narrow(1, 0, gamma).contiguous() :
                                                              torch::randint(vocab, {batch, gamma}, gpu_i32);
                auto                       uniforms     = torch::rand({batch, width}, gpu_f32);
                auto                       sample_mask  = torch::full({batch}, do_sample, gpu_f32.dtype(torch::kBool));
                std::vector<torch::Tensor> tokens, accepted;
                for (const auto& params : {full, compact}) {
                    tokens.push_back(torch::zeros({batch, width}, gpu_i32));
                    auto accept = torch::zeros({batch}, gpu_i32);
                    execRejectionSampling({draft_probs,
                                           draft_ids,
                                           uniforms,
                                           params.output_all_probs.value().reshape({batch, width, vocab}),
                                           params.token_ids,
                                           tokens.back(),
                                           accept,
                                           sample_mask,
                                           false,
                                           true});
                    accepted.push_back(torch::minimum(accept, lengths));
                }
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                EXPECT_TRUE(torch::equal(accepted[0], accepted[1]));
                auto accepted_cpu = accepted[0].cpu();
                for (int64_t request = 0; request < batch; ++request) {
                    const auto count  = accepted_cpu.data_ptr<int32_t>()[request];
                    const auto prefix = tokens[1][request].narrow(0, 0, count);
                    EXPECT_TRUE(torch::equal(tokens[0][request].narrow(0, 0, count), prefix));
                    EXPECT_TRUE(prefix.ge(0).logical_and(prefix.lt(vocab)).all().item<bool>());
                }
            }
        }
    }
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopK1) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 1, 1, 1});
    auto top_p_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(params);
    check_cuda_error();
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 7);
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopK) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 1, 0, 2});
    auto top_p_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(params);
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);
    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], (bool)expect_success[i]);
    }
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    assertTokenIn(output_token_ids_host, 17, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopP) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({0, 0, 0, 0});
    auto top_p_t      = pinnedFloatTensor({0.1, 0.1, 0.6, 0.8});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(params);
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);

    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    assertTokenIn(output_token_ids_host, 17, {7, 8, 5, 0, 4, 3});
    assertTokenIn(output_token_ids_host, 23, {7, 8, 5, 0, 4, 3, 9, 1});
}

TEST_F(CudaSamplerTest, testSeededTopPIsInvariantToBatchPosition) {
    constexpr int64_t vocab_size = 100;
    constexpr int64_t step       = 1;

    auto run = [&](int64_t batch_size) {
        auto cuda_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto cuda_int   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto pinned_int = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
        auto pinned_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);

        auto x           = torch::arange(vocab_size, cuda_float);
        auto logits_row  = torch::sin(x * 0.017) + torch::cos(x * 0.003);
        auto logits      = logits_row.repeat({batch_size, 1});
        auto token_ids   = torch::zeros({batch_size, step + 1}, cuda_int);
        auto lengths     = torch::full({batch_size}, step, cuda_int);
        auto top_k       = torch::zeros({batch_size}, pinned_int);
        auto top_p       = torch::full({batch_size}, 0.9, pinned_f32);
        auto temperature = torch::ones({batch_size}, pinned_f32);

        std::vector<at::Generator> generators;
        generators.reserve(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
            generator.set_current_seed(314159);
            generators.push_back(generator);
        }

        GreedyParams params({logits,
                             lengths,
                             lengths,
                             token_ids,
                             step,
                             top_k,
                             top_p,
                             temperature,
                             nullopt,
                             nullopt,
                             nullopt,
                             nullopt,
                             false,
                             nullopt,
                             nullopt,
                             nullopt,
                             nullopt,
                             generators});
        execSampleGreedy(params);
        check_cuda_error();
        return token_ids.select(1, step).cpu();
    };

    auto single  = run(1);
    auto batched = run(8);
    ASSERT_TRUE(batched.eq(single.item<int32_t>()).all().item<bool>())
        << "per-request seeded sampling changed with dynamic batch row";
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopKTopP) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 0, 0, 2});
    auto top_p_t      = pinnedFloatTensor({0.2, 0.2, 0.6, 0.6});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(params);
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);

    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    ASSERT_EQ(output_token_ids_host[5], 5);
    assertTokenIn(output_token_ids_host, 11, {2, 8});
    assertTokenIn(output_token_ids_host, 17, {7, 8, 5, 0, 4, 3});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testFlashinferKernelFailed) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0,    0,    0,     0.1, 0.2,  0.3,   0,     0,     0, 0.01,   0.987, 0.887, 0.99999, 0.1,
            0.2,  0.3,  0,     0,   0.99, 0.989, 0.221, 0,     0, 0.1,    0.2,   0.321, 0,       0.4432,
            0.44, 0.01, 0.221, 0,   0,    0.1,   0.2,   0.321, 0, 0.4432, 0.44,  0.01,
        },
        {(int64_t)batch_size, 10});
    size_t step             = 5;
    auto output_token_ids_t = cudaIntTensor({100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t      = pinnedIntTensor({1, 2, 2, 2});
    auto top_p_t      = pinnedFloatTensor({-1.0, -1.0, 0.6, 0.2});
    auto temperture_t = pinnedFloatTensor({1.0, 10.0, 1.0, 10.0});

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    GreedyParams params({
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperture_t,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        false,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        generator,
    });
    auto         greedy_output = execSampleGreedy(params);
    check_cuda_error();

    ASSERT_TRUE(greedy_output.success.defined());
    ASSERT_EQ(greedy_output.success.numel(), 4);

    auto         success_cpu = greedy_output.success.cpu();
    auto         success     = success_cpu.data_ptr<bool>();
    vector<bool> expect_success{false, false, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }
    auto output_token_ids_host = toHostInt(output_token_ids_t);
    assertTokenIn(output_token_ids_host, 17, {7, 8});
    assertTokenIn(output_token_ids_host, 23, {7, 8});
}

TEST_F(CudaSamplerTest, testBanRepeatNGram) {
    const auto vocab_size = 10;
    const auto batch_size = 4;
    const auto beam_width = 1;

    auto logits_t = cudaTensor(
        {
            0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1,
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step               = 8;
    auto   output_token_ids_t = cudaIntTensor(
        {
            0, 2, 3, 4, 5, 0, 0, 2, 0, 1, 2, 3, 3, 3, 1, 2, 3, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 9, 8, 6, 9, 8, 0, 0, 0, 0,
        },
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t     = cudaIntTensor({7, 7, 7, 4}, {4});
    auto no_repeat_ngram_size_t = cudaIntTensor({3, 4, 5, 2}, {4});

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    check_cuda_error();

    std::vector<uint64_t> output_ids_ptrs(batch_size);
    for (int i = 0; i < batch_size; i++) {
        output_ids_ptrs[i] = (uint64_t)(output_token_ids_t.data_ptr<int32_t>() + i * (step + 1));
    }
    auto output_ids_ptrs_t =
        torch::tensor(std::vector<int64_t>(output_ids_ptrs.begin(), output_ids_ptrs.end()), torch::kLong)
            .to(torch::kCUDA);

    tensorrt_llm::kernels::invokeBanRepeatNgram(logits_t.data_ptr<float>(),
                                                (int32_t const**)(output_ids_ptrs_t.data_ptr()),
                                                nullptr,  // finished_buf
                                                nullptr,  // parent_ids_buf
                                                nullptr,  // batch_slot
                                                sequence_lengths_t.data_ptr<int32_t>(),
                                                batch_size,
                                                beam_width,
                                                step,
                                                no_repeat_ngram_size_t.data_ptr<int32_t>(),
                                                vocab_size,
                                                step,
                                                stream);
    check_cuda_error();

    std::vector<int32_t> expcted_ban_token_ids = {3, 3, 1, 6};
    const auto           logits_cpu            = logits_t.cpu();
    for (int i = 0; i < batch_size; i++) {
        auto ban_id = expcted_ban_token_ids[i];
        for (int j = 0; j < vocab_size; j++) {
            if (j == ban_id) {
                EXPECT_EQ(logits_cpu[i][j].item<float>(), -INFINITY);
            } else {
                EXPECT_GT(logits_cpu[i][j].item<float>(), 0.0f);
            }
        }
    }
}

TEST_F(CudaSamplerTest, BatchTemperaturePenaltyFloatBatchExceedsBlock) {
    assertBatchTemperatureBeyondBlock(torch::kFloat32);
}

TEST_F(CudaSamplerTest, BatchTemperaturePenaltyHalfBatchExceedsBlock) {
    assertBatchTemperatureBeyondBlock(torch::kFloat16);
}

TEST_F(CudaSamplerTest, CompactHistoryTop1AllProbsLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80, 81920, 0);
}

TEST_F(CudaSamplerTest, CompactHistoryTop1NoAllProbsLong81920B80LeavesRngUnchanged) {
    assertCompactHistoryEquivalent(80, 81920, 0, /*with_all_probs=*/false);
}

TEST_F(CudaSamplerTest, CompactHistoryRandomTopKTopPPreservesRngAcrossShapes) {
    assertCompactHistoryEquivalent(80, 81920, 1);
    assertCompactHistoryEquivalent(5, 17, 1);
    assertCompactHistoryEquivalent(128, 31, 1);
}

TEST_F(CudaSamplerTest, CompactHistoryMixedDoSamplePreservesRngAcrossShapes) {
    assertCompactHistoryEquivalent(80, 81920, 2);
    assertCompactHistoryEquivalent(1, 9, 2);
    assertCompactHistoryEquivalent(80, 23, 2);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064Top1AllProbsLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80,
                                   81920,
                                   0,
                                   /*with_all_probs=*/true,
                                   /*vocab=*/200064,
                                   /*exact_probabilities=*/false);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064Top1NoAllProbsLong81920B80LeavesRngUnchanged) {
    assertCompactHistoryEquivalent(80, 81920, 0, /*with_all_probs=*/false, /*vocab=*/200064);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064RandomTopKTopPLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80,
                                   81920,
                                   1,
                                   /*with_all_probs=*/true,
                                   /*vocab=*/200064,
                                   /*exact_probabilities=*/false);
}

TEST_F(CudaSamplerTest, CompactHistoryRealVocab200064MixedDoSampleLong81920B80PreservesRng) {
    assertCompactHistoryEquivalent(80,
                                   81920,
                                   2,
                                   /*with_all_probs=*/true,
                                   /*vocab=*/200064,
                                   /*exact_probabilities=*/false);
}

// Opt-in bitwise diagnostic: intentionally fails for multi-CTA FP32 renorm,
// including full/full controls. Kept to reproduce the numerical-bound evidence;
// enabled regressions above enforce exact tokens/logits/RNG and bounded mass.
TEST_F(CudaSamplerTest, LargeVocabTopKRenormMatchesIndependentThresholdReference) {
    constexpr int64_t vocab = 200064;
    for (int64_t batch : {int64_t{3}, int64_t{15}, int64_t{80}}) {
        auto rows = torch::arange(batch, torch::kInt64).unsqueeze(1);
        auto cols = torch::arange(vocab, torch::kInt64).unsqueeze(0);
        auto logits = ((cols * 7 + rows * 3).remainder(31).to(torch::kFloat32) / 9.0f).cuda();
        auto probs = torch::softmax(logits, -1);
        const auto stream = at::cuda::getCurrentCUDAStream().stream();
        for (int64_t top_k : {int64_t{1}, int64_t{8}, int64_t{1 << 30}}) {
            SCOPED_TRACE(::testing::Message() << "batch=" << batch << " top_k=" << top_k);
            // FlashInfer retains every probability tied at the kth pivot.
            // Build the mathematical oracle independently of the radix kernel.
            auto filtered = probs;
            if (top_k < vocab) {
                auto values = std::get<0>(torch::topk(probs, top_k, -1));
                auto pivot = values.select(1, top_k - 1).unsqueeze(1);
                filtered = torch::where(probs.ge(pivot), probs, torch::zeros_like(probs));
            }
            auto expected = filtered / filtered.sum(-1, true);
            for (int repeat = 0; repeat < 12; ++repeat) {
                auto actual = torch::empty_like(probs);
                top_k_renorm_probs(probs, actual, std::nullopt, top_k, (int64_t)stream);
                ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
                ASSERT_TRUE(torch::allclose(actual, expected, 2e-5, 1e-8));
                ASSERT_TRUE(torch::equal(actual.gt(0), expected.gt(0)));
            }
        }
    }
}

TEST_F(CudaSamplerTest, testPenalty) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        },
        {(int64_t)batch_size, 10});
    size_t step               = 5;
    auto   output_token_ids_t = cudaIntTensor({2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0},
                                              {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
    auto cum_log_probs_t    = cudaTensor({-1.0, -2.0, -3.0, -3.0}, {4});

    auto top_k_t              = pinnedIntTensor({0, 0, 0, 0});
    auto top_p_t              = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t         = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto repetition_penalty_t = pinnedFloatTensor({2.4, 1.0, 1.0, 1.2});
    auto presence_penalty_t   = pinnedFloatTensor({0, 0.6, 0, 0.3});
    auto frequency_penalty_t  = pinnedFloatTensor({0, 0, 0.2, 0.1});

    auto output_all_probs_t = torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    std::vector<at::Generator> generator;
    for (int i = 0; i < batch_size; i++) {
        generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
        generator[i].set_current_seed(i + 1);
    }

    GreedyParams params({logits_t,
                         input_lengths_t,
                         sequence_lengths_t,
                         output_token_ids_t,
                         step,
                         top_k_t,
                         top_p_t,
                         temperture_t,
                         repetition_penalty_t,
                         nullopt,
                         cum_log_probs_t,
                         nullopt,
                         false,
                         output_all_probs_t,
                         presence_penalty_t,
                         frequency_penalty_t,
                         nullopt,
                         generator});
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    assertTokenIn(output_token_ids_host, 5, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 17, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    assertTokenIn(output_token_ids_host, 23, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0.0693098, 0.0990131, 0.100677,  0.075837,  0.0838128, 0.0926275, 0.102369,  0.113135,
                            0.125034,  0.138184,  0.0703223, 0.0921197, 0.0958792, 0.0769448, 0.0850372, 0.0939806,
                            0.103865,  0.114788,  0.126861,  0.140203,  0.080888,  0.12942,   0.110285,  0.0885056,
                            0.0978138, 0.108101,  0.11947,   0.0885056, 0.0885056, 0.0885056, 0.0715989, 0.0895156,
                            0.0837425, 0.0783417, 0.0865809, 0.0956867, 0.10575,   0.116872,  0.129164,  0.142748}),
        1e-3);
    auto cum_log_probs_host = toHostFloat(cum_log_probs_t);
    assertCumLogProbMatches(cum_log_probs_host,
                            output_all_probs_host,
                            output_token_ids_host,
                            {-1.0, -2.0, -3.0, -3.0},
                            batch_size,
                            step,
                            10);
}

TEST_F(CudaSamplerTest, testDoSample) {
    size_t batch_size = 4;
    auto   logits_t   = cudaTensor(
        {
            0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        },
        {(int64_t)batch_size, 10});
    size_t step               = 5;
    auto   output_token_ids_t = cudaIntTensor({2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0},
                                              {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cudaIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cudaIntTensor({-1, -1, -1, -1}, {4});
    auto cum_log_probs_t    = cudaTensor({-1.0, -2.0, -3.0, -3.0}, {4});

    auto top_k_t      = pinnedIntTensor({2, 2, 2, 2});
    auto top_p_t      = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperture_t = pinnedFloatTensor({2.0, 2.0, 4.0, 4.0});
    // do_sample: bool pinned tensor
    auto do_sample_t = torch::zeros({4}, torch::kBool).pin_memory();
    do_sample_t[0]   = false;
    do_sample_t[1]   = true;
    do_sample_t[2]   = false;
    do_sample_t[3]   = true;

    auto output_all_probs_t = torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    std::vector<at::Generator> generator;
    for (int i = 0; i < batch_size; i++) {
        generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
        generator[i].set_current_seed(i + 1);
    }

    GreedyParams params({logits_t,
                         input_lengths_t,
                         sequence_lengths_t,
                         output_token_ids_t,
                         step,
                         top_k_t,
                         top_p_t,
                         temperture_t,
                         nullopt,
                         nullopt,
                         cum_log_probs_t,
                         nullopt,
                         false,
                         output_all_probs_t,
                         nullopt,
                         nullopt,
                         do_sample_t,
                         generator});
    execSampleGreedy(params);
    check_cuda_error();

    auto output_token_ids_host = toHostInt(output_token_ids_t);
    assertTokenIn(output_token_ids_host, 5, {1, 2});
    assertTokenIn(output_token_ids_host, 11, {1, 2});
    assertTokenIn(output_token_ids_host, 17, {1, 2});
    assertTokenIn(output_token_ids_host, 23, {1, 2});

    auto output_all_probs_host = toHostFloat(output_all_probs_t);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.477515, 0.522485, 0, 0, 0, 0, 0, 0, 0,
                            0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.488752, 0.511248, 0, 0, 0, 0, 0, 0, 0}),
        1e-3);
    auto cum_log_probs_host = toHostFloat(cum_log_probs_t);
    assertCumLogProbMatches(cum_log_probs_host,
                            output_all_probs_host,
                            output_token_ids_host,
                            {-1.0, -2.0, -3.0, -3.0},
                            batch_size,
                            step,
                            10);
}
