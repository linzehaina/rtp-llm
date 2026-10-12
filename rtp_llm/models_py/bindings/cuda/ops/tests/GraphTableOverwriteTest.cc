#include "rtp_llm/cpp/cuda_graph/table_overwrite_plan.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <limits>

namespace rtp_llm {
namespace {

auto cudaIntOptions() {
    return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
}

CudaGraphPrepareFillParams fullFill(const torch::Tensor& dst) {
    CudaGraphPrepareFillParams fills;
    fills.region_count = 1;
    fills.regions[0] = {dst.data_ptr<int32_t>(), dst.numel(), 0};
    return fills;
}

bool plan(const torch::Tensor& src, const torch::Tensor& dst, const CudaGraphPrepareFillParams& fills,
          CudaGraphTableOverwriteParams& result,
          std::initializer_list<const torch::Tensor*> metadata = {}) {
    std::array<detail::GraphTableTensorPair, kMaxCudaGraphPrepareFillRegions> tables{};
    tables[0] = {&src, &dst};
    return detail::makeGraphTableOverwritePlan(tables, 1, fills, metadata, result);
}

TEST(GraphTableOverwriteTest, ActualKernelMatchesLegacyAndIndependentCpuBytes) {
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Includes real COMMIT width, page boundaries, ragged rows, and strided sources.
    for (const auto geometry : {std::array<int64_t, 4>{1, 1, 3, 5}, {3, 5, 4, 9},
                                {32, 625, 32, 708}, {32, 626, 32, 708},
                                {32, 627, 33, 711}, {159, 627, 161, 713}}) {
        const auto rows = geometry[0], cols = geometry[1], dst_rows = geometry[2], dst_cols = geometry[3];
        auto cpu = torch::arange(rows * (cols + 3), torch::TensorOptions().dtype(torch::kInt32))
                       .reshape({rows, cols + 3});
        cpu[0][0] = std::numeric_limits<int32_t>::max();
        auto src = cpu.to(torch::kCUDA).narrow(1, 0, cols);
        auto guarded = torch::full({dst_rows * dst_cols + 34}, -23, cudaIntOptions());
        auto dst = guarded.narrow(0, 17, dst_rows * dst_cols).view({dst_rows, dst_cols});
        auto old_dst = torch::full_like(dst, -29);
        auto expected = torch::zeros({dst_rows, dst_cols}, cpu.options());
        expected.narrow(0, 0, rows).narrow(1, 0, cols).copy_(cpu.narrow(1, 0, cols));
        CudaGraphTableOverwriteParams overwrite;
        ASSERT_TRUE(plan(src, dst, fullFill(dst), overwrite));
        invokeCudaGraphTableOverwrite(overwrite, stream);
        auto old_fill = fullFill(old_dst);
        invokeCudaGraphPrepareFill(old_fill, stream);
        FusedStridedCopyParams old_copy;
        old_copy.add(src.data_ptr(), old_dst.data_ptr(), rows, cols * sizeof(int32_t),
                     src.stride(0) * sizeof(int32_t), old_dst.stride(0) * sizeof(int32_t));
        invokeFusedStridedCopy(old_copy, stream);
        ASSERT_TRUE(torch::equal(dst.cpu(), expected));
        ASSERT_TRUE(torch::equal(dst, old_dst));
        auto guard_cpu = guarded.cpu();
        ASSERT_TRUE(guard_cpu.narrow(0, 0, 17).eq(-23).all().item<bool>());
        ASSERT_TRUE(guard_cpu.narrow(0, dst.numel() + 17, 17).eq(-23).all().item<bool>());
    }
}

TEST(GraphTableOverwriteTest, FillOnlyAndFreshValuesAcrossCudaGraphReplay) {
    auto src = torch::full({32, 626}, -31, cudaIntOptions());
    auto dst = torch::full({32, 708}, -23, cudaIntOptions());
    CudaGraphTableOverwriteParams overwrite;
    ASSERT_TRUE(plan(torch::Tensor(), dst, fullFill(dst), overwrite));
    invokeCudaGraphTableOverwrite(overwrite, at::cuda::getCurrentCUDAStream().stream());
    ASSERT_TRUE(dst.eq(0).all().item<bool>());
    ASSERT_TRUE(plan(src, dst, fullFill(dst), overwrite));
    auto capture_stream = at::cuda::getStreamFromPool();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(capture_stream.stream(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
    invokeCudaGraphTableOverwrite(overwrite, capture_stream.stream());
    ASSERT_EQ(cudaStreamEndCapture(capture_stream.stream(), &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    for (int32_t value : {-31, 0, std::numeric_limits<int32_t>::max(), 7}) {
        src.fill_(value);
        dst.fill_(-23);
        ASSERT_EQ(cudaGraphLaunch(exec, at::cuda::getCurrentCUDAStream().stream()), cudaSuccess);
        auto cpu = dst.cpu();
        ASSERT_TRUE(cpu.narrow(1, 0, 626).eq(value).all().item<bool>());
        ASSERT_TRUE(cpu.narrow(1, 626, 82).eq(0).all().item<bool>());
    }
    ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST(GraphTableOverwriteTest, RejectsNonTableFillCapacityDtypeAndEveryWriteAliasBeforeLaunch) {
    auto src = torch::full({3, 5}, 7, cudaIntOptions());
    auto dst = torch::full({4, 9}, -23, cudaIntOptions());
    CudaGraphTableOverwriteParams overwrite;
    auto fill = fullFill(dst);
    ASSERT_TRUE(plan(src, dst, fill, overwrite));
    auto extra_fill = fill;
    auto meta = torch::zeros({4}, cudaIntOptions());
    extra_fill.regions[extra_fill.region_count++] = {meta.data_ptr<int32_t>(), 4, 0};
    ASSERT_FALSE(plan(src, dst, extra_fill, overwrite));
    ASSERT_EQ(overwrite.count, 0);
    auto partial_fill = fill;
    partial_fill.regions[0].count--;
    ASSERT_FALSE(plan(src, dst, partial_fill, overwrite));
    auto nonzero_fill = fill;
    nonzero_fill.regions[0].value = 1;
    ASSERT_FALSE(plan(src, dst, nonzero_fill, overwrite));
    auto larger = torch::zeros({5, 10}, cudaIntOptions());
    ASSERT_FALSE(plan(larger, dst, fill, overwrite));
    ASSERT_FALSE(plan(src.to(torch::kInt64), dst, fill, overwrite));
    ASSERT_FALSE(plan(src.transpose(0, 1), dst, fill, overwrite));
    auto alias = dst.narrow(0, 0, 3).narrow(1, 0, 5);
    ASSERT_FALSE(plan(alias, dst, fill, overwrite));
    auto dst_meta = dst.reshape({-1}).narrow(0, 4, 2);
    ASSERT_FALSE(plan(src, dst, fill, overwrite, {&dst_meta}));
    auto src_meta = src.reshape({-1}).narrow(0, 3, 2);
    ASSERT_FALSE(plan(src, dst, fill, overwrite, {&src_meta}));
    std::array<detail::GraphTableTensorPair, kMaxCudaGraphPrepareFillRegions> tables{};
    tables[0] = {&src, &dst};
    tables[1] = {&src, &dst};
    auto duplicate_fill = fill;
    duplicate_fill.regions[duplicate_fill.region_count++] = fill.regions[0];
    ASSERT_FALSE(detail::makeGraphTableOverwritePlan(tables, 2, duplicate_fill, {}, overwrite));
    ASSERT_TRUE(dst.eq(-23).all().item<bool>());
}

TEST(GraphTableOverwriteTest, MultipleTablesHonorSourceReadyAcrossStreams) {
    auto producer = at::cuda::getStreamFromPool();
    auto consumer = at::cuda::getStreamFromPool();
    auto src1 = torch::zeros({3, 5}, cudaIntOptions());
    auto src2 = torch::zeros({3, 7}, cudaIntOptions());
    auto dst1 = torch::full({4, 9}, -23, cudaIntOptions());
    auto dst2 = torch::full({4, 11}, -23, cudaIntOptions());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaEvent_t ready;
    ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
    {
        c10::cuda::CUDAStreamGuard guard(producer);
        src1.fill_(31);
        src2.fill_(-7);
        ASSERT_EQ(cudaEventRecord(ready, producer.stream()), cudaSuccess);
    }
    std::array<detail::GraphTableTensorPair, kMaxCudaGraphPrepareFillRegions> tables{};
    tables[0] = {&src1, &dst1};
    tables[1] = {&src2, &dst2};
    auto fills = fullFill(dst1);
    fills.regions[fills.region_count++] = fullFill(dst2).regions[0];
    CudaGraphTableOverwriteParams overwrite;
    ASSERT_TRUE(detail::makeGraphTableOverwritePlan(tables, 2, fills, {}, overwrite));
    ASSERT_EQ(cudaStreamWaitEvent(consumer.stream(), ready, 0), cudaSuccess);
    invokeCudaGraphTableOverwrite(overwrite, consumer.stream());
    ASSERT_EQ(cudaStreamSynchronize(consumer.stream()), cudaSuccess);
    ASSERT_TRUE(dst1.narrow(0, 0, 3).narrow(1, 0, 5).eq(31).all().item<bool>());
    ASSERT_TRUE(dst2.narrow(0, 0, 3).narrow(1, 0, 7).eq(-7).all().item<bool>());
    ASSERT_TRUE(dst1.narrow(0, 3, 1).eq(0).all().item<bool>());
    ASSERT_TRUE(dst2.narrow(1, 7, 4).eq(0).all().item<bool>());
    ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
}

}  // namespace
}  // namespace rtp_llm
