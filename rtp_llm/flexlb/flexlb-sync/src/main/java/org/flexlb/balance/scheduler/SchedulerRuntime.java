package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;

/** Owns scheduler maintenance, metrics traversal, and ordered shutdown. */
@Component
public final class SchedulerRuntime implements RequestTaskScheduler {

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    private final DeliveryMetricsReporter deliveryReporter;
    private final RequestSchedulerReporter requestReporter;
    private final DefaultBatchDispatcher dispatcher;
    private final ConfigService config;
    private final LongSupplier clock;

    private final EngineCancelChannel cancelChannel;
    private final ScheduledThreadPoolExecutor cleanupExecutor =
            new ScheduledThreadPoolExecutor(2,
                    Thread.ofPlatform().daemon().name("flexlb-request-cleanup-", 1).factory());

    @Override
    public Cancellation scheduleTimeout(Runnable work, long delayMs) {
        var scheduled = cleanupExecutor.schedule(Objects.requireNonNull(work, "work"),
                delayMs, TimeUnit.MILLISECONDS);
        return () -> scheduled.cancel(false);
    }

    void startDeliveryCleanup(RequestContext.DeliveryClaim claim) {
        new DeliveryCleanupTask(claim, cancelChannel, cleanupExecutor,
                work -> executeRequestWork(claim.item.ctx(), work)).start();
    }

    private final ExecutorService requestWorkers = Executors.newFixedThreadPool(
            Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8),
            Thread.ofPlatform().daemon().name("request-work-", 1).factory());
    private final ExecutorService requestRecovery = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().daemon().name("request-work-recovery-", 1).factory());
    // Never run request code while holding this monitor. Context -> workLifecycle is the only edge.
    private final Object workLifecycle = new Object();
    private boolean acceptingWork = true;
    private long pendingWork;
    private final ResponseCompletionExecutor responseCompletions;
    private final ExpirationTimer timer;

    RequestRepository requests() { return requests; }

    /** Offloads effects; Context and endpoint owners arbitrate concurrent work, not executor order. */
    @Override
    public void executeRequestWork(RequestContext context, Runnable work) {
        Objects.requireNonNull(work, "work");
        try (var registration = registerRequestWork(context)) { registration.submit(work); }
    }

    /** Register before publishing state that obliges a later effect or local cleanup. */
    WorkRegistration registerRequestWork(RequestContext context) {
        WorkRegistration registration = tryRegisterRequestWork(context);
        checkState(registration != null, "request work is closed");
        return registration;
    }

    WorkRegistration tryRegisterRequestWork(RequestContext context) {
        Objects.requireNonNull(context, "context");
        synchronized (workLifecycle) {
            if (!acceptingWork) { return null; }
            pendingWork++;
        }
        return new WorkRegistration(context);
    }

    /** Counts both the submitting source and its optional task, including the handoff between them. */
    final class WorkRegistration implements AutoCloseable {
        private final RequestContext context;
        private int owners = 1;
        private boolean submitted;
        private boolean sourceClosed;
        private boolean started;

        private WorkRegistration(RequestContext context) { this.context = context; }

        void submit(Runnable work) {
            Objects.requireNonNull(work, "work");
            context.requireOutsideContextLock("request work submission");
            synchronized (this) {
                checkState(!sourceClosed && !submitted, "request work already handed off");
                submitted = true;
                owners++;
            }
            Runnable task = () -> run(work);
            try { requestWorkers.execute(task); }
            catch (RuntimeException rejected) {
                try { requestRecovery.execute(task); }
                catch (RuntimeException recoveryRejected) {
                    // Accepted terminal/publication owners cannot be dropped or run on an RPC callback.
                    recordFailure(Failures.append(rejected, recoveryRejected));
                    try { Thread.ofPlatform().daemon().name("request-work-emergency").start(task); }
                    catch (RuntimeException | Error failedToStart) {
                        recordFailure(failedToStart);
                        release();
                        throw failedToStart;
                    }
                }
            }
        }

        private void run(Runnable work) {
            synchronized (this) {
                if (started) { return; }
                started = true;
            }
            try { work.run(); }
            catch (Throwable failure) {
                recordFailure(failure);
                try { Logger.error("Request work failed: request_id={}", context.getRequestId(), failure); }
                catch (Throwable ignored) { }
            } finally { release(); }
        }

        private void release() {
            synchronized (this) { if (--owners != 0) { return; } }
            synchronized (workLifecycle) {
                pendingWork--;
                if (pendingWork == 0) { workLifecycle.notifyAll(); }
            }
        }

        @Override
        public void close() {
            synchronized (this) {
                if (sourceClosed) { return; }
                sourceClosed = true;
            }
            release();
        }
    }

    void awaitRequestWork() {
        boolean interrupted = false;
        synchronized (workLifecycle) {
            while (pendingWork != 0) {
                try { workLifecycle.wait(); }
                catch (InterruptedException ignored) { interrupted = true; }
            }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }

    void closeRequestWork() {
        synchronized (workLifecycle) {
            // Nested work remains admissible until every accepted source and task has left.
            awaitRequestWork();
            acceptingWork = false;
        }
        requestWorkers.shutdown();
        requestRecovery.shutdown();
        boolean interrupted = false;
        while (!requestWorkers.isTerminated() || !requestRecovery.isTerminated()) {
            try {
                requestWorkers.awaitTermination(1, TimeUnit.DAYS);
                requestRecovery.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }
    ResponseCompletionExecutor.CompletionRegistration registerResponseCompletion() {
        return responseCompletions.tryRegister();
    }
    ExpirationTimer.Scheduling deadlines() { return timer.scheduling(); }
    DeliveryMetricsReporter deliveryReporter() { return deliveryReporter; }
    RequestSchedulerReporter requestReporter() { return requestReporter; }

    private final Object schedulerLock = new Object();
    private SchedulerLifecycle schedulerLifecycle;
    private Throwable failure;

    boolean isAccepting() { return !requests.isStopping() && !requests.isClosed(); }

    /** 只停收。已注册请求继续运行；shutdown 才执行整套关闭流程。 */
    public void stopAccepting() {
        requests.stopAccepting();
    }

    /** 后台清理、准入或续接失败在此汇总；停服时统一报告，不执行 Future 监听或关闭流程。 */
    void recordFailure(Throwable cause) {
        synchronized (schedulerLock) { failure = Failures.append(failure, cause); }
    }

    void initializeScheduler(SchedulerLifecycle schedulerLifecycle) {
        synchronized (schedulerLock) {
            checkState(isAccepting(), "scheduler is closed");
            checkState(this.schedulerLifecycle == null, "scheduler already initialized");
            this.schedulerLifecycle = Objects.requireNonNull(schedulerLifecycle, "schedulerLifecycle");
        }
    }

    @Autowired
    SchedulerRuntime(
            RequestRepository requests,
            EndpointRegistry endpoints,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            DefaultBatchDispatcher dispatcher, ConfigService config,
            EngineCancelChannel cancelChannel) {
        this(requests, endpoints, deliveryReporter, requestReporter, dispatcher, config, cancelChannel, System::currentTimeMillis);
    }

    SchedulerRuntime(RequestRepository requests, EndpointRegistry endpoints, DeliveryMetricsReporter deliveryReporter,
                     RequestSchedulerReporter requestReporter, DefaultBatchDispatcher dispatcher,
                     ConfigService config,
                     EngineCancelChannel cancelChannel, LongSupplier clock) {
        this.cancelChannel = Objects.requireNonNull(cancelChannel, "cancelChannel");
        cleanupExecutor.setRemoveOnCancelPolicy(true);
        cleanupExecutor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        cleanupExecutor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter, "deliveryReporter");
        this.requestReporter = Objects.requireNonNull(requestReporter, "requestReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.config = Objects.requireNonNull(config, "config");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.responseCompletions = new ResponseCompletionExecutor(config.loadBalanceConfig().getInternalRuntime().getBatchDispatchCompletionThreads());
        this.timer = new ExpirationTimer(requests);
    }

    @Scheduled(fixedRate = EXPIRATION_MAINTENANCE_INTERVAL_MS)
    void maintainExpiration() {
        if (requests.isClosed()) { return; }
        long ttlMs = config.loadBalanceConfig().getWorkerRegistry().getHealth().getStatusStaleAfterMs();
        long nowMs = clock.getAsLong();
        Throwable failure = Failures.run(null,
                () -> requests.expireTerminalRecords(subtractSaturated(nowMs, ttlMs)));
        failure = Failures.run(failure,
                () -> endpoints.evictExpiredOrphans(ttlMs, requests::retainsIdentity));
        Failures.rethrow(failure, "expiration maintenance failed");
    }

    private static long subtractSaturated(long value, long decrement) {
        try { return Math.subtractExact(value, decrement); }
        catch (ArithmeticException underflow) { return Long.MIN_VALUE; }
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (requests.isClosed()) {
            return;
        }
        reportSchedulerInflight();
        reportEndpoints("Prefill", endpoints::snapshotPrefillEndpoints,
                endpoint -> endpoint.reportBatchMetrics(deliveryReporter),
                (address, endpoint) -> requestReporter.reportPrefillQueueDepth(address, endpoint.queuedRequestCount()));
        reportEndpoints("Decode", endpoints::snapshotDecodeEndpoints,
                endpoint -> endpoint.reportBatchMetrics(deliveryReporter),
                (address, endpoint) -> endpoint.reportAdmissionMetrics(requestReporter));
    }

    private void reportSchedulerInflight() {
        try {
            deliveryReporter.reportSchedulerInflight(requests.liveRequestCount(), requests.oldestLiveRequestAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to report scheduler inflight metrics", failure);
        }
    }

    private static <T> void reportEndpoints(
            String role, Supplier<Map<String, T>> snapshot,
            Consumer<T> batchMetrics, BiConsumer<String, T> admissionMetrics) {
        Map<String, T> endpoints;
        try {
            endpoints = snapshot.get();
        } catch (RuntimeException failure) {
            warnIsolated("Failed to snapshot " + role + " endpoints for metrics", failure);
            return;
        }
        for (Map.Entry<String, T> entry : endpoints.entrySet()) {
            try {
                batchMetrics.accept(entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated("Failed to report " + role + " endpoint metrics: endpoint=" + entry.getKey(), failure);
            }
            try {
                admissionMetrics.accept(entry.getKey(), entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated("Failed to report " + role + " admission metrics: endpoint=" + entry.getKey(), failure);
            }
        }
    }

    private static void warnIsolated(
            String message, RuntimeException failure) {
        try {
            Logger.warn(message, failure);
        } catch (Throwable ignored) {
            // Telemetry must never couple otherwise independent metric leaves.
        }
    }

    private SchedulerLifecycle initializedSchedulerLifecycle() {
        synchronized (schedulerLock) {
            return schedulerLifecycle;
        }
    }

    private void closeRequestExecutors() {
        Throwable failure = null;
        try { closeRequestWork(); } catch (Throwable cause) { failure = cause; }
        try { responseCompletions.close(); } catch (Throwable cause) { failure = Failures.append(failure, cause); }
        cleanupExecutor.shutdown();
        boolean interrupted = false;
        while (!cleanupExecutor.isTerminated()) {
            try { cleanupExecutor.awaitTermination(1, TimeUnit.DAYS); }
            catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
        Failures.rethrow(failure, "request executors failed to close");
    }

    /** Spring 停服入口：停止接收、等待请求资源结算，再关闭共享执行设施。 */
    @PreDestroy
    void shutdown() {
        stopAccepting();
        if (!requests.closeRegistration()) { return; }
        var lifecycle = initializedSchedulerLifecycle();
        if (lifecycle == null) { lifecycle = new SchedulerLifecycle(requests, null); }
        Runnable[] steps = {
                lifecycle::closePlacement,
                lifecycle::awaitAdmissionMutations,
                lifecycle::abandonOutstandingDeliveries,
                dispatcher::shutdownAndAwait,
                lifecycle::awaitDeliveryCleanup,
                endpoints::close,
                timer::close,
                this::awaitRequestWork,
                lifecycle::closeOutstandingAndTerminalize,
                this::closeRequestExecutors
        };
        Throwable firstFailure = null;
        for (Runnable step : steps) {
            try { step.run(); }
            catch (Throwable failure) {
                if (firstFailure == null) { firstFailure = failure; }
                else if (failure != firstFailure) { firstFailure.addSuppressed(failure); }
            }
        }
        if (requests.liveRequestCount() != 0) {
            firstFailure = Failures.append(firstFailure,
                    new IllegalStateException("Unsettled requests at shutdown: " + requests.liveRequestCount()));
        }
        synchronized (schedulerLock) { firstFailure = Failures.append(failure, firstFailure); }
        Failures.rethrow(firstFailure, "Scheduler shutdown failed");
    }

}
