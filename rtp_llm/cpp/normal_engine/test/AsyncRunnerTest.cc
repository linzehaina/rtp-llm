#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <ATen/record_function.h>
#include "gtest/gtest.h"
#include "torch/torch.h"

#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
namespace {

std::atomic<int>* g_record_function_callback_count = nullptr;

// Release a deliberately gated producer at the existing host-barrier entry.
// This is test-only observation: no sleep decides whether streamWait is safe.
// The old bare event.block never enters this scope and returns with the producer
// still gated; tests release it unconditionally before draining/destruction.
thread_local std::function<void()>* g_sync_entry_action = nullptr;

std::unique_ptr<at::ObserverContext> releaseProducerAtSyncEntry(const at::RecordFunction& fn) {
    if (g_sync_entry_action && std::strcmp(fn.name(), "async_runner.sync") == 0) {
        (*g_sync_entry_action)();
    }
    return nullptr;
}

class ControlledProducer {
public:
    bool enterAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        started_ = true;
        cv_.notify_all();
        // A watchdog bounds test failure/cleanup; elapsed time is not an oracle.
        return cv_.wait_for(lock, std::chrono::seconds(10), [&] { return released_; });
    }
    bool waitUntilStarted() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(10), [&] { return started_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool started_ = false, released_ = false;
};

class ScopedSyncEntryAction {
public:
    explicit ScopedSyncEntryAction(std::function<void()> action): action_(std::move(action)) {
        g_sync_entry_action = &action_;
        handle_ = at::addThreadLocalCallback(at::RecordFunctionCallback(&releaseProducerAtSyncEntry));
    }
    ~ScopedSyncEntryAction() {
        at::removeCallback(handle_);
        g_sync_entry_action = nullptr;
    }
private:
    std::function<void()> action_;
    at::CallbackHandle handle_;
};

std::unique_ptr<at::ObserverContext> countRecordFunctionCallback(const at::RecordFunction&) {
    if (g_record_function_callback_count != nullptr) {
        g_record_function_callback_count->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

}  // namespace

class AsyncRunnerTest: public ::testing::Test {
protected:
    torch::Stream makeStream() {
        return cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/true);
    }

    torch::Stream currentStream() {
        return cuda_graph::graphGetCurrentStream();
    }

    void checkDelayedEventPublication(bool previous_event_completed) {
        ControlledProducer producer;
        std::atomic<bool> published{false};
        auto stream = makeStream();
        auto dst = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        currentStream().synchronize();
        AsyncRunner runner(stream);
        if (previous_event_completed) {
            runner.launch([dst] { dst.fill_(1); });
            runner.sync(currentStream());
            currentStream().synchronize();
        }
        runner.launch([&producer, &published, dst] {
            if (!producer.enterAndWait()) throw std::runtime_error("producer gate watchdog expired");
            dst.fill_(7);
            published.store(true, std::memory_order_release);
            // workerLoop records event_ only after this task body returns.
        });
        if (!producer.waitUntilStarted()) {
            producer.release();
            (void)runner.joinAndDrain();
            FAIL() << "producer did not enter controlled gate";
        }
        {
            ScopedSyncEntryAction action([&producer] { producer.release(); });
            runner.streamWait(currentStream());
        }
        const bool publication_observed = published.load(std::memory_order_acquire);
        // Capture the consumer result before releasing the old implementation's
        // gated task; otherwise cleanup could accidentally repair a stale wait.
        auto observed = dst.clone();
        currentStream().synchronize();
        producer.release();
        EXPECT_FALSE(runner.joinAndDrain());
        EXPECT_TRUE(publication_observed);
        EXPECT_TRUE(torch::equal(observed.cpu(), torch::full({64}, 7, torch::kFloat32)));
    }
};

TEST_F(AsyncRunnerTest, StreamWaitPublishesFirstTaskBeforeOrderingConsumer) {
    checkDelayedEventPublication(false);
}

TEST_F(AsyncRunnerTest, StreamWaitPublishesRerecordedTaskNotPreviousCompletedEvent) {
    checkDelayedEventPublication(true);
}

TEST_F(AsyncRunnerTest, StreamWaitPropagatesPublishedFailureAndAllowsRecovery) {
    AsyncRunner runner(makeStream());
    runner.launch([] { throw std::runtime_error("streamWait task failure"); });
    {
        // Establish that the worker published its error without consuming it.
        std::unique_lock<std::mutex> lock(runner.mutex_);
        ASSERT_TRUE(runner.cv_done_.wait_for(lock, std::chrono::seconds(10), [&] { return runner.task_done_; }));
    }
    EXPECT_THROW(runner.streamWait(currentStream()), std::runtime_error);
    auto dst = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    currentStream().synchronize();
    // If the old implementation did not propagate the exception, drain consumes
    // it so the regression still cleans up and can check subsequent recovery.
    (void)runner.joinAndDrain();
    runner.launch([dst] { dst.fill_(9); });
    runner.streamWait(currentStream());
    EXPECT_TRUE(torch::equal(dst.cpu(), torch::full({64}, 9, torch::kFloat32)));
}

TEST_F(AsyncRunnerTest, LaunchAndSyncBasic) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());

    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, SequentialLaunches) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    constexpr int    kIterations = 10;
    for (int i = 0; i < kIterations; i++) {
        runner.launch([&counter] { counter.fetch_add(1); });
    }
    runner.sync(currentStream());

    EXPECT_EQ(counter.load(), kIterations);
}

TEST_F(AsyncRunnerTest, ExecutionOrder) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::vector<int> order;
    std::mutex       mu;

    for (int i = 0; i < 5; i++) {
        runner.launch([&order, &mu, i] {
            std::lock_guard<std::mutex> lk(mu);
            order.push_back(i);
        });
    }
    runner.sync(currentStream());

    ASSERT_EQ(order.size(), 5u);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(order[i], i);
    }
}

TEST_F(AsyncRunnerTest, CudaTensorWork) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    auto src = torch::ones({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto dst = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Inputs were initialized on the caller stream, while the worker uses a
    // private stream. Order that initialization before its first tensor read.
    auto inputs_ready = cuda_graph::makeGraphEvent();
    inputs_ready.record(currentStream());
    inputs_ready.block(stream);
    runner.launch([&src, &dst] { dst.copy_(src); });
    runner.sync(currentStream());

    auto result = dst.cpu();
    EXPECT_TRUE(torch::equal(result, torch::ones({64}, torch::kFloat32)));
}

TEST_F(AsyncRunnerTest, DoesNotPropagateRecordFunctionCallbacksToWorker) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int>  callback_count{0};
    std::atomic<bool> allow_worker_ops{false};
    std::atomic<bool> worker_task_started{false};
    g_record_function_callback_count = &callback_count;
    auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(&countRecordFunctionCallback));

    {
        at::RecordFunction main_record(at::RecordScope::FUNCTION);
        main_record.before("async_runner.main_thread_positive_control");
        main_record.end();
    }
    EXPECT_GT(callback_count.load(std::memory_order_relaxed), 0);
    callback_count.store(0, std::memory_order_relaxed);

    runner.launch([&allow_worker_ops, &worker_task_started] {
        worker_task_started.store(true, std::memory_order_release);
        while (!allow_worker_ops.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto src = torch::ones({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto dst = src.narrow(0, 0, 32).clone();
        (void)dst;
    });

    while (!worker_task_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    callback_count.store(0, std::memory_order_relaxed);
    at::removeCallback(handle);
    allow_worker_ops.store(true, std::memory_order_release);
    runner.sync(currentStream());

    g_record_function_callback_count = nullptr;
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 0);
}

TEST_F(AsyncRunnerTest, DestructorJoinsThread) {
    auto             stream = makeStream();
    std::atomic<int> counter{0};

    {
        AsyncRunner runner(stream);
        runner.launch([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            counter.fetch_add(1);
        });
    }
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, SyncWithoutLaunch) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.sync(currentStream());
}

TEST_F(AsyncRunnerTest, RethrowsWorkerExceptionFromSync) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.launch([] { throw std::runtime_error("async failure"); });
    EXPECT_THROW(runner.sync(currentStream()), std::runtime_error);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, RethrowsWorkerExceptionFromNextLaunch) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.launch([] { throw std::runtime_error("async failure"); });
    EXPECT_THROW(runner.launch([] {}), std::runtime_error);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, LaunchSyncInterleavedStress) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    constexpr int    kIterations = 100;

    for (int i = 0; i < kIterations; i++) {
        runner.launch([&counter] { counter.fetch_add(1); });
        runner.sync(currentStream());
        EXPECT_EQ(counter.load(), i + 1);
    }
}

TEST_F(AsyncRunnerTest, JoinAndDrainWithoutLaunch) {
    AsyncRunner runner(makeStream());
    EXPECT_FALSE(runner.joinAndDrain());
}

TEST_F(AsyncRunnerTest, JoinAndDrainPreservesTaskErrorAndAllowsRecovery) {
    AsyncRunner runner(makeStream());
    auto        dst = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    currentStream().synchronize();
    runner.launch([dst] {
        dst.fill_(7);
        throw std::runtime_error("failure after GPU submission");
    });
    auto error = runner.joinAndDrain();
    ASSERT_TRUE(error);
    EXPECT_THROW(std::rethrow_exception(error), std::runtime_error);
    EXPECT_TRUE(torch::equal(dst.cpu(), torch::full({64}, 7, torch::kFloat32)));
    EXPECT_FALSE(runner.joinAndDrain());
    runner.launch([dst] { dst.fill_(9); });
    EXPECT_FALSE(runner.joinAndDrain());
    EXPECT_TRUE(torch::equal(dst.cpu(), torch::full({64}, 9, torch::kFloat32)));
}

TEST_F(AsyncRunnerTest, JoinAndDrainRejectsSelfJoin) {
    AsyncRunner runner(makeStream());
    runner.launch([&runner] { (void)runner.joinAndDrain(); });
    auto error = runner.joinAndDrain();
    ASSERT_TRUE(error);
    EXPECT_THROW(std::rethrow_exception(error), std::logic_error);
}

TEST_F(AsyncRunnerTest, UnstartedFailureCleanupIsExactlyOnce) {
    AsyncRunner::Task task;
    int               cleanups = 0;
    task.on_unstarted_failure  = [&](std::exception_ptr error) {
        EXPECT_TRUE(error);
        ++cleanups;
    };
    auto error = std::make_exception_ptr(std::runtime_error("injected stream setup failure"));
    AsyncRunner::finishUnstartedTaskFailure(task, error);
    AsyncRunner::finishUnstartedTaskFailure(task, error);
    EXPECT_EQ(cleanups, 1);
    EXPECT_THROW(std::rethrow_exception(error), std::runtime_error);
}

TEST_F(AsyncRunnerTest, StartedTaskDoesNotRunUnstartedCleanup) {
    AsyncRunner      runner(makeStream());
    std::atomic<int> cleanups{0};
    runner.launch([] { throw std::runtime_error("task body failure"); }, [&](std::exception_ptr) { ++cleanups; });
    EXPECT_TRUE(runner.joinAndDrain());
    EXPECT_EQ(cleanups.load(), 0);
}

}  // namespace rtp_llm
