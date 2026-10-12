#include "rtp_llm/models_py/bindings/core/SamplingSeedOffset.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>

namespace rtp_llm {
namespace {

using Pair = std::pair<torch::Tensor, torch::Tensor>;

// Independent legacy oracle: two allocations and two uploads, including the
// original per-row default/undefined-generator fallback and uint64 bit cast.
Pair legacy(const std::vector<at::Generator>& generators, int64_t rows, TensorHolder* holder) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto seeds = torch::empty({rows}, options);
    auto offsets = torch::empty({rows}, options);
    for (int64_t i = 0; i < rows; ++i) {
        auto generator = i < static_cast<int64_t>(generators.size()) && generators[i].defined()
                             ? std::make_optional(generators[i]) : std::nullopt;
        auto [seed, offset] = get_seed_and_offset(32, generator);
        seeds.data_ptr<int64_t>()[i] = static_cast<int64_t>(seed);
        offsets.data_ptr<int64_t>()[i] = static_cast<int64_t>(offset);
    }
    auto seed_d = seeds.to(torch::kCUDA, true).contiguous();
    auto offset_d = offsets.to(torch::kCUDA, true).contiguous();
    if (holder) {
        holder->hold_host(seeds);
        holder->hold_host(offsets);
    }
    return {seed_d, offset_d};
}

at::Generator makeGenerator(uint64_t seed) {
    auto gen = at::make_generator<at::CUDAGeneratorImpl>();
    gen.set_current_seed(seed);
    return gen;
}

struct RngSnapshot {
    std::vector<at::Generator> generators;
    std::vector<torch::Tensor> states;
    explicit RngSnapshot(std::vector<at::Generator> unique): generators(std::move(unique)) {
        generators.push_back(at::cuda::detail::getDefaultCUDAGenerator());
        for (auto& gen : generators) states.push_back(gen.get_state().clone());
    }
    void restore() {
        for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
    }
    ~RngSnapshot() { restore(); }
};

TEST(SamplingSeedOffsetTest, DenseRowsBytesGeneratorStateAndNextRandomDraw) {
    for (int64_t rows : {0, 1, 3, 32, 160, 224}) {
        for (int mode = 0; mode < 4; ++mode) {
            SCOPED_TRACE(std::to_string(rows) + "/" + std::to_string(mode));
            std::vector<at::Generator> unique = {
                makeGenerator(0), makeGenerator(uint64_t{1} << 63), makeGenerator(UINT64_MAX)};
            unique[1].set_offset((uint64_t{1} << 32) + 4096);
            std::vector<at::Generator> mapping;
            for (int64_t i = 0; i < rows; ++i) {
                if (mode == 1 && i >= rows / 2) break;  // shorter vector falls back to default
                if (mode == 0) break;
                mapping.push_back(mode == 1 && i % 3 == 0 ? at::Generator{}
                                  : unique[mode == 2 ? 1 : i % unique.size()]);
            }
            RngSnapshot before(unique);
            TensorHolder baseline_holder;
            auto expected = legacy(mapping, rows, &baseline_holder);
            std::vector<torch::Tensor> after;
            std::vector<torch::Tensor> next;
            for (auto& gen : before.generators) {
                after.push_back(gen.get_state().clone());
                next.push_back(torch::rand({17}, gen, torch::TensorOptions().device(torch::kCUDA)));
            }
            before.restore();
            TensorHolder holder;
            auto actual = detail::makeSamplingSeedOffsetTensors(mapping, rows, 32, &holder);
            ASSERT_EQ(holder.tensors.size(), 1);
            ASSERT_EQ(holder.tensors[0].is_pinned(), baseline_holder.tensors[0].is_pinned());
            ASSERT_EQ(holder.tensors[0].sizes(), torch::IntArrayRef({2, rows}));
            ASSERT_EQ(actual.first.sizes(), torch::IntArrayRef({rows}));
            ASSERT_EQ(actual.second.sizes(), torch::IntArrayRef({rows}));
            ASSERT_EQ(actual.first.scalar_type(), torch::kInt64);
            ASSERT_TRUE(actual.first.is_cuda());
            ASSERT_TRUE(actual.first.is_contiguous());
            ASSERT_TRUE(actual.second.is_contiguous());
            ASSERT_EQ(actual.first.storage().unsafeGetStorageImpl(), actual.second.storage().unsafeGetStorageImpl());
            if (rows > 0) {
                ASSERT_EQ(actual.second.storage_offset(), rows);
            }
            ASSERT_TRUE(torch::equal(expected.first.cpu(), actual.first.cpu()));
            ASSERT_TRUE(torch::equal(expected.second.cpu(), actual.second.cpu()));
            for (size_t i = 0; i < before.generators.size(); ++i) {
                ASSERT_TRUE(torch::equal(after[i], before.generators[i].get_state()));
                auto draw = torch::rand({17}, before.generators[i], torch::TensorOptions().device(torch::kCUDA));
                ASSERT_TRUE(torch::equal(next[i], draw));
            }
        }
    }
}

TEST(SamplingSeedOffsetTest, NonDefaultStreamNullHolderAndReleasedOwnerRemainValid) {
    auto stream = at::cuda::getStreamFromPool();
    auto other_stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto generator = makeGenerator(UINT64_MAX);
    RngSnapshot before({generator});
    std::vector<Pair> expected, actual;
    TensorHolder holder;
    for (int round = 0; round < 24; ++round) {
        auto rows = std::vector<int64_t>{1, 3, 160, 224}[round % 4];
        std::vector<at::Generator> mapping(rows, generator);
        auto oracle = legacy(mapping, rows, nullptr);
        expected.emplace_back(oracle.first.cpu(), oracle.second.cpu());
    }
    auto expected_state = generator.get_state().clone();
    before.restore();
    for (int round = 0; round < 24; ++round) {
        auto rows = std::vector<int64_t>{1, 3, 160, 224}[round % 4];
        std::vector<at::Generator> mapping(rows, generator);
        auto copy_stream = round % 2 ? other_stream : stream;
        c10::cuda::CUDAStreamGuard copy_guard(copy_stream);
        ASSERT_EQ(cudaLaunchHostFunc(copy_stream.stream(), [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }, nullptr), cudaSuccess);
        actual.push_back(detail::makeSamplingSeedOffsetTensors(mapping, rows, 32, round % 2 ? &holder : nullptr));
        // Release through all retired sets while H2D may still be queued. The
        // pinned allocator's source-event protection must also cover null holders.
        for (int i = 0; i < 4; ++i) holder.release();
        for (int i = 0; i < 12; ++i) {
            auto pressure = torch::empty({2, rows}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
            pressure.fill_(-23);
        }
    }
    ASSERT_TRUE(torch::equal(expected_state, generator.get_state()));
    ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(other_stream.stream()), cudaSuccess);
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(torch::equal(expected[i].first, actual[i].first.cpu()));
        ASSERT_TRUE(torch::equal(expected[i].second, actual[i].second.cpu()));
    }
}

TEST(SamplingSeedOffsetTest, ActualSamplingConsumersPreserveDenseAndCompactResults) {
    for (int64_t rows : {1, 3, 32, 160}) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto probs = torch::rand({rows, 1024}, options);
        probs.div_(probs.sum(-1, true));
        auto generator = makeGenerator(UINT64_MAX);
        std::vector<at::Generator> mapping(rows, generator);
        RngSnapshot before({generator});
        auto expected = legacy(mapping, rows, nullptr);
        before.restore();
        auto actual = detail::makeSamplingSeedOffsetTensors(mapping, rows, 32, nullptr);
        for (bool compact : {false, true}) {
            auto indices = torch::arange(0, rows, 2, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto live_probs = compact ? probs.index_select(0, indices) : probs;
            auto live_rows = live_probs.size(0);
            auto reference = compact ? Pair{expected.first.index_select(0, indices),
                                             expected.second.index_select(0, indices)} : expected;
            auto candidate = compact ? Pair{actual.first.index_select(0, indices),
                                             actual.second.index_select(0, indices)} : actual;
            for (int mode = 0; mode < 3; ++mode) {
                Pair results[2];
                for (int arm = 0; arm < 2; ++arm) {
                    auto arrays = arm ? candidate : reference;
                    auto output = torch::empty({live_rows}, options.dtype(torch::kInt32));
                    auto valid = torch::empty({live_rows}, options.dtype(torch::kBool));
                    auto stream = at::cuda::getCurrentCUDAStream().stream();
                    if (mode == 0)
                        top_k_sampling_from_probs(live_probs, output, valid, std::nullopt, std::nullopt, 4,
                                                  true, arrays.first, 0, arrays.second, 0, int64_t(stream));
                    else if (mode == 1)
                        top_p_sampling_from_probs(live_probs, output, valid, std::nullopt, std::nullopt, 0.83,
                                                  true, arrays.first, 0, arrays.second, 0, int64_t(stream));
                    else
                        top_k_top_p_sampling_from_probs(live_probs, output, valid, std::nullopt, std::nullopt, 32,
                                                        std::nullopt, 0.83, true, arrays.first, 0,
                                                        arrays.second, 0, int64_t(stream));
                    results[arm] = {output.cpu(), valid.cpu()};
                }
                ASSERT_TRUE(torch::equal(results[0].first, results[1].first));
                ASSERT_TRUE(torch::equal(results[0].second, results[1].second));
                ASSERT_TRUE(results[0].second.all().item<bool>());
            }
        }
    }
}

TEST(SamplingSeedOffsetTest, DISABLED_InterleavedT160SubmissionAndConsumerBenchmark) {
    auto generator = makeGenerator(42);
    std::vector<at::Generator> mapping(160, generator);
    RngSnapshot before({generator});
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto probs = torch::rand({160, 200064}, options);
    probs.div_(probs.sum(-1, true));
    auto output = torch::empty({160}, options.dtype(torch::kInt32));
    auto valid = torch::empty({160}, options.dtype(torch::kBool));
    TensorHolder holders[2];
    std::vector<double> cpu_us[2], chain_us[2];
    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (int iteration = -50; iteration < 500; ++iteration) {
        auto state = generator.get_state().clone();
        for (int turn = 0; turn < 2; ++turn) {
            int arm = (turn + (iteration & 1)) & 1;
            generator.set_state(state);
            ASSERT_EQ(cudaEventRecord(start, stream), cudaSuccess);
            auto begin = std::chrono::steady_clock::now();
            auto pair = arm ? detail::makeSamplingSeedOffsetTensors(mapping, 160, 32, &holders[arm])
                            : legacy(mapping, 160, &holders[arm]);
            top_k_sampling_from_probs(probs, output, valid, std::nullopt, std::nullopt, 4, true,
                                      pair.first, 0, pair.second, 0, int64_t(stream));
            holders[arm].release();
            auto end = std::chrono::steady_clock::now();
            ASSERT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
            float elapsed_ms;
            ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);
            if (iteration >= 0) {
                cpu_us[arm].push_back(std::chrono::duration<double, std::micro>(end - begin).count());
                chain_us[arm].push_back(elapsed_ms * 1000.0);
            }
        }
    }
    for (int arm = 0; arm < 2; ++arm) {
        for (size_t i = 0; i < cpu_us[arm].size(); ++i)
            std::cout << "SEED_OFFSET_SAMPLE arm=" << arm << " pair=" << i
                      << " cpu_us=" << cpu_us[arm][i] << " chain_us=" << chain_us[arm][i] << '\n';
        std::sort(cpu_us[arm].begin(), cpu_us[arm].end());
        std::sort(chain_us[arm].begin(), chain_us[arm].end());
        std::cout << "SEED_OFFSET_BENCH arm=" << arm << " pairs=500 cpu_median_us="
                  << cpu_us[arm][250] << " cpu_p95_us=" << cpu_us[arm][475]
                  << " chain_median_us=" << chain_us[arm][250]
                  << " chain_p95_us=" << chain_us[arm][475] << std::endl;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

}  // namespace
}  // namespace rtp_llm
