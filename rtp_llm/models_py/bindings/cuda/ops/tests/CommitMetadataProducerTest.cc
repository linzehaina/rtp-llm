#include "rtp_llm/models_py/bindings/cuda/kernels/commit_metadata_producer.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <thread>
#include <vector>

namespace rtp_llm {
namespace {

constexpr int kGuard = 17;
constexpr int32_t kCanary = -213456789;
auto gpuInt() { return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA); }

torch::Tensor upload(const std::vector<int32_t>& values) {
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kCUDA);
}

// Independent CPU oracle uses wide totals, then preserves the low 32 bits.
// It does not invoke any GPU metadata builder or mirror the per-row kernel scan.
int32_t lowInt32(int64_t value) {
    const uint32_t low = static_cast<uint32_t>(static_cast<uint64_t>(value));
    int32_t result;
    std::memcpy(&result, &low, sizeof(result));
    return result;
}

struct Guarded {
    torch::Tensor owner, data;
    std::vector<int32_t> original;
    explicit Guarded(int64_t size, int32_t salt = 0) : original(size + 2 * kGuard, kCanary) {
        for (int64_t i = 0; i < size; ++i) original[kGuard + i] = 101 + salt + int32_t(i);
        owner = upload(original);
        data = owner.narrow(0, kGuard, size);
    }
    std::vector<int32_t> initial() const {
        return {original.begin() + kGuard, original.end() - kGuard};
    }
    void expect(const std::vector<int32_t>& payload) const {
        ASSERT_EQ(payload.size(), static_cast<size_t>(data.numel()));
        auto expected = original;
        std::copy(payload.begin(), payload.end(), expected.begin() + kGuard);
        auto cpu = owner.cpu().contiguous();
        ASSERT_EQ(cpu.numel(), static_cast<int64_t>(expected.size()));
        ASSERT_EQ(std::memcmp(cpu.data_ptr<int32_t>(), expected.data(),
                              expected.size() * sizeof(int32_t)), 0);
    }
};

struct Table {
    int rows, cols, stride, capacity, width;
    std::vector<int32_t> source_cpu;
    torch::Tensor source_owner, source;
    Guarded output;
    bool fill_only;
    Table(int batch, int cap, int live_cols, int dst_cols, int salt, bool only_fill = false) :
        rows(batch), cols(live_cols), stride(live_cols + 3), capacity(cap), width(dst_cols),
        source_cpu(int64_t(batch) * stride), output(int64_t(cap) * dst_cols, salt),
        fill_only(only_fill) {
        for (int64_t i = 0; i < static_cast<int64_t>(source_cpu.size()); ++i)
            source_cpu[i] = lowInt32(100000LL * (salt + 1) + i);
        if (!source_cpu.empty()) source_cpu[0] = std::numeric_limits<int32_t>::max();
        source_owner = upload(source_cpu).reshape({batch, stride});
        source = source_owner.narrow(1, 0, cols);
    }
    CudaGraphTableOverwriteTask task() const {
        return {fill_only || rows == 0 ? nullptr : source.data_ptr<int32_t>(),
                output.data.data_ptr<int32_t>(), rows, cols, stride, capacity, width};
    }
    std::vector<int32_t> expected() const {
        std::vector<int32_t> result(int64_t(capacity) * width, 0);
        if (!fill_only) for (int row = 0; row < rows; ++row)
            std::copy_n(source_cpu.begin() + int64_t(row) * stride, cols,
                        result.begin() + int64_t(row) * width);
        return result;
    }
};

struct Case {
    int batch, capacity;
    std::vector<int32_t> lengths, prefixes, plus;
    torch::Tensor source_lengths, source_prefixes, source_plus;
    Guarded input, prefix, cu, cu_kv, plus1, sequence, decode_cu;
    std::vector<Table> tables;
    CudaGraphCommitMetadataParams params;
    Case(int b, int c, bool explicit_plus, bool optional_tails = true, bool static_five = false,
         int table_count = 3) :
        batch(b), capacity(c), lengths(b), prefixes(b), plus(b),
        input(c, 11), prefix(c, 23), cu(c + 1, 37), cu_kv(c + 1, 41),
        plus1(c, 53), sequence(c, 67), decode_cu(c + 1, 79) {
        constexpr int32_t boundary[] = {0, 127, 128, 129, 255, 79999, 80000, 80256};
        for (int row = 0; row < batch; ++row) {
            lengths[row] = static_five ? 5 : std::array<int32_t, 5>{0, 1, 5, 2, 0}[row % 5];
            prefixes[row] = boundary[row % 8];
            plus[row] = prefixes[row] + 17; // intentionally not prefix+1
        }
        source_lengths = upload(lengths);
        source_prefixes = upload(prefixes);
        source_plus = upload(plus);
        params.input_lengths = batch ? source_lengths.data_ptr<int32_t>() : nullptr;
        params.prefix_lengths = batch ? source_prefixes.data_ptr<int32_t>() : nullptr;
        params.sequence_lengths_plus_1 = explicit_plus && batch ? source_plus.data_ptr<int32_t>() : nullptr;
        params.output_input_lengths = input.data.data_ptr<int32_t>();
        params.output_prefix_lengths = prefix.data.data_ptr<int32_t>();
        params.output_cu_seqlens = cu.data.data_ptr<int32_t>();
        params.output_cu_kv_seqlens = cu_kv.data.data_ptr<int32_t>();
        params.output_sequence_lengths_plus_1 = plus1.data.data_ptr<int32_t>();
        params.output_sequence_lengths = optional_tails ? sequence.data.data_ptr<int32_t>() : nullptr;
        params.output_decode_cu_seqlens = optional_tails ? decode_cu.data.data_ptr<int32_t>() : nullptr;
        params.batch_size = batch;
        params.batch_capacity = capacity;
        params.last_valid = 17; // checked independently of the fresh live sums
        tables.reserve(table_count);
        for (int t = 0; t < table_count; ++t) {
            tables.emplace_back(batch, capacity, 626 + t, 708 + t, t, t == 2);
            params.tables.tasks[t] = tables.back().task();
        }
        params.tables.count = table_count;
    }

    void check() const {
        auto expected_input = input.initial(), expected_prefix = prefix.initial();
        auto expected_cu = cu.initial(), expected_kv = cu_kv.initial();
        auto expected_plus = plus1.initial(), expected_seq = sequence.initial();
        auto expected_decode = decode_cu.initial();
        int64_t token_total = 0, kv_total = 0;
        expected_cu[0] = expected_kv[0] = 0;
        for (int row = 0; row < batch; ++row) {
            expected_input[row] = lengths[row];
            expected_prefix[row] = prefixes[row];
            expected_plus[row] = params.sequence_lengths_plus_1 ? plus[row] : lowInt32(int64_t(prefixes[row]) + 1);
            token_total += lengths[row];
            kv_total += int64_t(lengths[row]) + prefixes[row];
            expected_cu[row + 1] = lowInt32(token_total);
            expected_kv[row + 1] = lowInt32(kv_total);
        }
        for (int row = batch; row < capacity; ++row) {
            expected_input[row] = expected_prefix[row] = expected_plus[row] = 0;
            if (params.output_sequence_lengths) expected_seq[row] = 0;
        }
        for (int row = batch + 1; row <= capacity; ++row) {
            expected_cu[row] = params.last_valid;
            if (params.output_decode_cu_seqlens) expected_decode[row] = params.last_valid;
        }
        input.expect(expected_input); prefix.expect(expected_prefix); plus1.expect(expected_plus);
        cu.expect(expected_cu); cu_kv.expect(expected_kv);
        sequence.expect(expected_seq); decode_cu.expect(expected_decode);
        for (const auto& table : tables) table.output.expect(table.expected());
    }
};

TEST(CommitMetadataProducerTest, IndependentBytesRaggedStaticFivePaddedAndEmpty) {
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (const auto shape : {std::array<int, 2>{32, 32}, {31, 32}, {32, 33}, {1, 4}, {0, 4}, {0, 0}}) {
        for (bool explicit_plus : {false, true}) for (bool tails : {false, true})
            for (bool five : {false, true}) {
                SCOPED_TRACE(::testing::Message() << shape[0] << "/" << shape[1]
                             << " plus=" << explicit_plus << " tails=" << tails << " five=" << five);
                Case test(shape[0], shape[1], explicit_plus, tails, five);
                invokeCudaGraphCommitMetadataProducer(test.params, stream);
                test.check();
            }
    }
}

TEST(CommitMetadataProducerTest, MetadataOnlyLargeCapacityAndInt32Wrap) {
    Case test(3, 513, false, true, false, 0); // metadata threads stride beyond 256
    test.lengths = {std::numeric_limits<int32_t>::max(), 1, 5};
    test.prefixes = {std::numeric_limits<int32_t>::max(), 128, 80000};
    test.source_lengths.copy_(torch::tensor(test.lengths, torch::kInt32));
    test.source_prefixes.copy_(torch::tensor(test.prefixes, torch::kInt32));
    invokeCudaGraphCommitMetadataProducer(test.params, at::cuda::getCurrentCUDAStream().stream());
    test.check();
}

TEST(CommitMetadataProducerTest, CapturedSingleLaunchReadsFreshSourcesEveryReplay) {
    auto capture = at::cuda::getStreamFromPool();
    Case test(32, 33, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(capture.stream(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
    invokeCudaGraphCommitMetadataProducer(test.params, capture.stream());
    ASSERT_EQ(cudaStreamEndCapture(capture.stream(), &graph), cudaSuccess);
    size_t nodes = 0;
    ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &nodes), cudaSuccess);
    ASSERT_EQ(nodes, 1); // one producer, no allocation/query/fill launches in invoke
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    for (int round = 0; round < 4; ++round) {
        for (int row = 0; row < test.batch; ++row) {
            test.lengths[row] = (row + round) % 6;
            test.prefixes[row] = 79999 + row + round * 128;
            test.plus[row] = test.prefixes[row] + 3 + round;
        }
        test.source_lengths.copy_(torch::tensor(test.lengths, torch::kInt32));
        test.source_prefixes.copy_(torch::tensor(test.prefixes, torch::kInt32));
        test.source_plus.copy_(torch::tensor(test.plus, torch::kInt32));
        for (auto& table : test.tables) {
            for (auto& value : table.source_cpu) value = round ? -round * 31 : std::numeric_limits<int32_t>::max();
            table.source_owner.copy_(torch::tensor(table.source_cpu, torch::kInt32).reshape({table.rows, table.stride}));
        }
        ASSERT_EQ(cudaGraphLaunch(executable, at::cuda::getCurrentCUDAStream().stream()), cudaSuccess);
        test.check();
    }
    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST(CommitMetadataProducerTest, RejectsInvalidHostDescriptorsBeforeLaunch) {
    Case test(1, 3, false);
    auto p = test.params;
    p.batch_size = 4;
    EXPECT_ANY_THROW(invokeCudaGraphCommitMetadataProducer(p, nullptr));
    p = test.params; p.tables.count = kMaxCudaGraphPrepareFillRegions + 1;
    EXPECT_ANY_THROW(invokeCudaGraphCommitMetadataProducer(p, nullptr));
    p = test.params; p.input_lengths = nullptr;
    EXPECT_ANY_THROW(invokeCudaGraphCommitMetadataProducer(p, nullptr));
    p = test.params; p.output_cu_kv_seqlens = nullptr;
    EXPECT_ANY_THROW(invokeCudaGraphCommitMetadataProducer(p, nullptr));
    p = test.params; p.tables.tasks[0].src_stride = 1;
    EXPECT_ANY_THROW(invokeCudaGraphCommitMetadataProducer(p, nullptr));
    p = test.params; p.tables.tasks[0].dst_rows = std::numeric_limits<int64_t>::max();
    EXPECT_ANY_THROW(invokeCudaGraphCommitMetadataProducer(p, nullptr));
    // No failed descriptor may launch a fill that destroys a captured old value.
    test.input.expect(test.input.initial());
    test.tables[0].output.expect(test.tables[0].output.initial());
}

TEST(CommitMetadataProducerTest, DelayedTwoStreamsSourceProducerConsumerAndOwnerPressure) {
    auto source_stream = at::cuda::getStreamFromPool();
    auto commit_stream = at::cuda::getStreamFromPool();
    Case test(31, 32, false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaEvent_t ready = nullptr, committed = nullptr, consumed = nullptr;
    ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&committed, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming), cudaSuccess);
    {
        c10::cuda::CUDAStreamGuard guard(source_stream);
        ASSERT_EQ(cudaLaunchHostFunc(source_stream.stream(), [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }, nullptr), cudaSuccess);
        test.source_lengths.fill_(5);
        test.source_prefixes.fill_(80000);
        std::fill(test.lengths.begin(), test.lengths.end(), 5);
        std::fill(test.prefixes.begin(), test.prefixes.end(), 80000);
        for (auto& table : test.tables) {
            table.source_owner.fill_(37);
            std::fill(table.source_cpu.begin(), table.source_cpu.end(), 37);
        }
        ASSERT_EQ(cudaEventRecord(ready, source_stream.stream()), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamWaitEvent(commit_stream.stream(), ready, 0), cudaSuccess);
    invokeCudaGraphCommitMetadataProducer(test.params, commit_stream.stream());
    ASSERT_EQ(cudaEventRecord(committed, commit_stream.stream()), cudaSuccess);
    // Caller owners stay retained until consumer completion; native raw pointers
    // do not provide implicit storage lifetime or cross-stream synchronization.
    std::vector<torch::Tensor> observed;
    {
        c10::cuda::CUDAStreamGuard guard(source_stream);
        ASSERT_EQ(cudaStreamWaitEvent(source_stream.stream(), committed, 0), cudaSuccess);
        observed.push_back(test.input.data.clone());
        observed.push_back(test.prefix.data.clone());
        observed.push_back(test.cu.data.clone());
        observed.push_back(test.cu_kv.data.clone());
        observed.push_back(test.plus1.data.clone());
        for (const auto& table : test.tables) observed.push_back(table.output.data.clone());
        ASSERT_EQ(cudaEventRecord(consumed, source_stream.stream()), cudaSuccess);
        // Same-sized fresh allocations while both streams can still be queued.
        std::vector<torch::Tensor> pressure;
        for (int i = 0; i < 32; ++i) {
            auto slab = torch::empty_like(test.tables[0].source_owner);
            slab.fill_(-31);
            pressure.push_back(std::move(slab));
        }
    }
    ASSERT_EQ(cudaEventSynchronize(consumed), cudaSuccess);
    test.check();
    const std::array<const Guarded*, 5> metadata = {&test.input, &test.prefix, &test.cu, &test.cu_kv, &test.plus1};
    for (size_t i = 0; i < metadata.size(); ++i)
        EXPECT_TRUE(torch::equal(observed[i].cpu(), metadata[i]->data.cpu()));
    for (size_t i = 0; i < test.tables.size(); ++i)
        EXPECT_TRUE(torch::equal(observed[i + 5].cpu(), test.tables[i].output.data.cpu()));
    ASSERT_EQ(cudaStreamSynchronize(source_stream.stream()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(commit_stream.stream()), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(committed), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(consumed), cudaSuccess);
}


// Disabled: parent explicitly runs after correctness gates. Fixed GPU sources /
// destinations exclude mandatory table H2D, allocation, wrapper routing checks,
// source ownership setup and model consumption. GPU graph elapsed and native
// host submission/POD-plan construction are reported separately.
TEST(CommitMetadataProducerTest, DISABLED_PairedFullCommitGraphAndSubmissionBenchmark) {
    constexpr int rounds = 9, pairs = 40, graph_repeats = 32, warmups = 10;
    struct Sample {
        bool explicit_plus;
        int round, pair, arm;
        double graph_device_us, graph_host_submit_us, native_host_submit_us;
        double plan_cpu_us;
    };
    std::vector<Sample> samples;
    cudaEvent_t start = nullptr, stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (bool explicit_plus : {false, true}) {
        Case old(32, 32, explicit_plus, false, true, 1);
        Case fresh(32, 32, explicit_plus, false, true, 1);
        auto old_cu = torch::empty({33}, gpuInt());
        auto old_kv = torch::empty({33}, gpuInt());
        auto derived_plus = torch::empty({32}, gpuInt());
        torch::Tensor no_padding;
        auto copies = [&]() {
            FusedD2DCopyParams p;
            p.add(old.source_lengths.data_ptr(), old.input.data.data_ptr(), 32 * sizeof(int32_t));
            p.add(old.source_prefixes.data_ptr(), old.prefix.data.data_ptr(), 32 * sizeof(int32_t));
            p.add(old_cu.data_ptr(), old.cu.data.data_ptr(), 33 * sizeof(int32_t));
            p.add(old_kv.data_ptr(), old.cu_kv.data.data_ptr(), 33 * sizeof(int32_t));
            p.add(explicit_plus ? old.source_plus.data_ptr() : derived_plus.data_ptr(),
                  old.plus1.data.data_ptr(), 32 * sizeof(int32_t));
            return p;
        };
        auto copy_params = copies();
        auto launch_old = [&](cudaStream_t launch_stream) {
            invokeBuildAttentionInputMetadata(old.source_lengths, old.source_prefixes,
                                              old_cu, old_kv, no_padding, launch_stream);
            if (!explicit_plus) at::add_out(derived_plus, old.source_prefixes, 1);
            invokeFusedCopy(copy_params, launch_stream);
            invokeCudaGraphTableOverwrite(old.params.tables, launch_stream);
        };
        cudaGraph_t graphs[2] = {nullptr, nullptr};
        cudaGraphExec_t executable[2] = {nullptr, nullptr};
        auto capture = at::cuda::getStreamFromPool();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int arm = 0; arm < 2; ++arm) {
            c10::cuda::CUDAStreamGuard guard(capture);
            ASSERT_EQ(cudaStreamBeginCapture(capture.stream(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
            if (arm) invokeCudaGraphCommitMetadataProducer(fresh.params, capture.stream());
            else launch_old(capture.stream());
            ASSERT_EQ(cudaStreamEndCapture(capture.stream(), &graphs[arm]), cudaSuccess);
            ASSERT_EQ(cudaGraphInstantiate(&executable[arm], graphs[arm], nullptr, nullptr, 0), cudaSuccess);
        }
        for (int round = -1; round < rounds; ++round) {
            for (int pair = 0; pair < (round < 0 ? warmups : pairs); ++pair) {
                for (int turn = 0; turn < 2; ++turn) {
                    const int arm = (pair + turn + (round < 0 ? 0 : round)) & 1;
                    ASSERT_EQ(cudaEventRecord(start, stream), cudaSuccess);
                    auto begin = std::chrono::steady_clock::now();
                    for (int repeat = 0; repeat < graph_repeats; ++repeat)
                        ASSERT_EQ(cudaGraphLaunch(executable[arm], stream), cudaSuccess);
                    auto submitted = std::chrono::steady_clock::now();
                    ASSERT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
                    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
                    float graph_ms = 0;
                    ASSERT_EQ(cudaEventElapsedTime(&graph_ms, start, stop), cudaSuccess);

                    // Native launches include old builder checks/query and the
                    // optional real ATen prefix add, unlike graph replay.
                    auto native_begin = std::chrono::steady_clock::now();
                    if (arm) invokeCudaGraphCommitMetadataProducer(fresh.params, stream);
                    else launch_old(stream);
                    auto native_end = std::chrono::steady_clock::now();
                    ASSERT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
                    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

                    auto plan_begin = std::chrono::steady_clock::now();
                    if (arm) {
                        auto plan = fresh.params;
                        asm volatile("" : : "m"(plan) : "memory");
                    } else {
                        auto plan = copies();
                        asm volatile("" : : "m"(plan) : "memory");
                    }
                    auto plan_end = std::chrono::steady_clock::now();
                    if (round >= 0) samples.push_back({
                        explicit_plus, round, pair, arm, graph_ms * 1000.0 / graph_repeats,
                        std::chrono::duration<double, std::micro>(submitted - begin).count() / graph_repeats,
                        std::chrono::duration<double, std::micro>(native_end - native_begin).count(),
                        std::chrono::duration<double, std::micro>(plan_end - plan_begin).count()});
                }
            }
        }
        old.check();
        fresh.check();
        for (int arm = 0; arm < 2; ++arm) {
            ASSERT_EQ(cudaGraphExecDestroy(executable[arm]), cudaSuccess);
            ASSERT_EQ(cudaGraphDestroy(graphs[arm]), cudaSuccess);
        }
    }
    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
    std::cout << "{\"type\":\"commit_bench_metadata\",\"B\":32,\"C\":32,\"T\":160,"
                 "\"tables\":1,\"source_cols\":626,\"destination_cols\":708,"
                 "\"rounds\":9,\"pairs_per_round\":40,\"graph_repeats\":32,"
                 "\"source_table_H2D_included\":false,"
                 "\"CPU_plan_scope\":\"POD_only_excludes_wrapper_validation\"}" << std::endl;
    for (const auto& s : samples)
        std::cout << "{\"type\":\"commit_bench_sample\",\"explicit_plus\":" << s.explicit_plus
                  << ",\"round\":" << s.round << ",\"pair\":" << s.pair << ",\"arm\":" << s.arm
                  << ",\"graph_device_us\":" << s.graph_device_us
                  << ",\"graph_host_submit_us\":" << s.graph_host_submit_us
                  << ",\"native_host_submit_us\":" << s.native_host_submit_us
                  << ",\"plan_cpu_us\":" << s.plan_cpu_us << "}" << std::endl;
    for (bool explicit_plus : {false, true}) for (int arm = 0; arm < 2; ++arm) {
        std::array<std::vector<double>, 4> values;
        for (const auto& s : samples) if (s.explicit_plus == explicit_plus && s.arm == arm) {
            values[0].push_back(s.graph_device_us); values[1].push_back(s.graph_host_submit_us);
            values[2].push_back(s.native_host_submit_us); values[3].push_back(s.plan_cpu_us);
        }
        const char* names[] = {"graph_device", "graph_host_submit", "native_host_submit", "plan_cpu"};
        for (int metric = 0; metric < 4; ++metric) {
            auto& v = values[metric]; std::sort(v.begin(), v.end());
            std::cout << "{\"type\":\"commit_bench_summary\",\"explicit_plus\":" << explicit_plus
                      << ",\"arm\":" << arm << ",\"metric\":\"" << names[metric]
                      << "\",\"n\":" << v.size() << ",\"median_us\":" << v[v.size() / 2]
                      << ",\"p95_us\":" << v[v.size() * 95 / 100] << "}" << std::endl;
        }
    }
}

} // namespace
} // namespace rtp_llm
