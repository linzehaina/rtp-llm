load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "speculative_sampler_test",
    srcs = ["SpeculativeSamplerTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "mtp_batch_stream_processor_test",
    srcs = glob([
        "MtpBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "mtp_executor_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    args = ["--gtest_filter=-MtpExecutorTest.DSparkDispatchMetricsPublishesBeforeSameStepConsumption:MtpExecutorTest.DSparkDispatchMetricsPreviousLaunchErrorPreservedAfterCleanup:MtpExecutorTest.DSparkDispatchMetricsCollectionFailureDoesNotPreventAcceptedWorker:MtpExecutorTest.NonDSparkDispatchMetricsCollectsBeforeAsyncPublication:MtpExecutorTest.DSparkMetricsFaultThroughProcessDrainsWorkerAndReaderBeforeLeaseRelease"],
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "RTP_LLM_STREAM_ASYNC": "0",
        "RTP_LLM_DROP_BROAD_SYNC": "0",
    },
    exec_properties = {'gpu':'H20'},
)

# Cached runtime flags require separate processes. Legacy mock-sampler tests
# produce CPU tensors; async lifecycle tests use the actual GPU publication path.
sh_test(
    name = "mtp_executor_async_test",
    srcs = ["//:cc_test_wrapper.sh"],
    data = [":mtp_executor_test_bin"],
    args = [
        "$(location :mtp_executor_test_bin)",
        "--gtest_filter=-MtpExecutorTest.DSparkDispatchMetricsSyncCollectsBeforePublication:MtpExecutorTest.testSingleBatchDecode:MtpExecutorTest.testMultiBatchDecode",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "RTP_LLM_STREAM_ASYNC": "1",
        "RTP_LLM_DROP_BROAD_SYNC": "1",
    },
    exec_properties = {"gpu": "H20"},
)
