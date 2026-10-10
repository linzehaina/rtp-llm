package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkState;

/**
 * Runs asynchronous frontend completion tasks on dedicated workers. External Future
 * mutations use completeNow on their caller to preserve synchronous return semantics.
 * Owns execution registration and drain. After request cleanup, the external lifecycle
 * owner closes it. Completion callbacks do not own executor shutdown. The scheduler
 * owns request arbitration, lock checks, deadlines and ACK reporting.
 */
final class ResponseCompletionExecutor implements AutoCloseable {

    /**
     * Reserves a close obligation before submission. The caller closes unused registrations;
     * submit or completeNow releases accepted registrations after execution, including failure.
     * Contains no request state or response policy.
     */
    static final class CompletionRegistration {
        private final ResponseCompletionExecutor owner;
        private final AtomicBoolean closed = new AtomicBoolean();

        CompletionRegistration(ResponseCompletionExecutor owner) {
            this.owner = Objects.requireNonNull(owner);
        }

        void submit(BooleanSupplier operation) { owner.submit(this, operation); }

        boolean completeNow(BooleanSupplier operation) { return owner.completeNow(this, operation); }

        void close() {
            if (closed.compareAndSet(false, true)) { owner.exitCompletion(this); }
        }
    }

    private final ThreadPoolExecutor completionWorkers;
    private final ExecutorService rejectedTaskWorker = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().daemon().name("response-completion-recovery-", 1).factory());

    /** Null while accepting registrations; otherwise the shared final close result. */
    private CompletableFuture<Throwable> closeResult;

    /** Also serves as the monitor protecting registrations and closeResult. */
    private final Set<CompletionRegistration> registrations =
            Collections.newSetFromMap(new IdentityHashMap<>());

    ResponseCompletionExecutor(int configuredWorkers) {
        completionWorkers = new ThreadPoolExecutor(
                configuredWorkers,
                configuredWorkers,
                0L,
                TimeUnit.MILLISECONDS,
                // Queue completions so a busy completion executor never runs client callbacks
                // inline on a decision thread. RequestContext owns request lifetime; the
                // executor owns only these in-flight frontend completions.
                new LinkedBlockingQueue<>(),
                Thread.ofPlatform().daemon().name("response-completion-", 0).factory(),
                new ThreadPoolExecutor.AbortPolicy());
        completionWorkers.prestartAllCoreThreads();
    }

    CompletionRegistration tryRegister() {
        synchronized (registrations) {
            if (closeResult != null) { return null; }
            var registration = new CompletionRegistration(this);
            registrations.add(registration);
            return registration;
        }
    }

    private void exitCompletion(CompletionRegistration registration) {
        synchronized (registrations) {
            registrations.remove(registration);
            if (registrations.isEmpty()) { registrations.notifyAll(); }
        }
    }

    private void requireOwnedRegistration(CompletionRegistration registration) {
        checkState(registration.owner == this, "completion registration belongs to another executor");
    }

    /** Runs a scheduler-owned completion operation, without interpreting request facts. */
    void submit(CompletionRegistration registration, BooleanSupplier operation) {
        try {
            requireOwnedRegistration(registration);
            try {
                completionWorkers.execute(() -> completeNow(registration, operation));
            } catch (RejectedExecutionException rejection) {
                // An accepted operation must not run client callbacks on its submitter.
                rejectedTaskWorker.execute(() -> completeNow(registration, operation));
            }
        } catch (RuntimeException | Error failure) {
            registration.close();
            throw failure;
        }
    }

    /** External Future mutations stay synchronous; every execution releases its registration. */
    boolean completeNow(CompletionRegistration registration, BooleanSupplier operation) {
        try {
            requireOwnedRegistration(registration);
            return operation.getAsBoolean();
        } finally {
            registration.close();
        }
    }

    /** Stops registration, drains accepted completions and closes workers. */
    @Override
    public void close() {
        boolean interrupted = false;
        try {
            synchronized (registrations) {
                if (closeResult == null) {
                    closeResult = new CompletableFuture<>();
                }
                while (!registrations.isEmpty()) {
                    try {
                        registrations.wait();
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                // Drained workers no longer need this monitor. Serialize final
                // closure so concurrent callers receive the same result.
                if (!closeResult.isDone()) {
                    finishClose();
                }
                Failures.rethrow(closeResult.join(), "response completion executor close failed");
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void finishClose() {
        boolean interrupted = false;
        Throwable failure = null;
        try {
            failure = Failures.run(null, completionWorkers::shutdown);
            failure = Failures.run(failure, rejectedTaskWorker::shutdown);
            while (failure == null && (!completionWorkers.isTerminated() || !rejectedTaskWorker.isTerminated())) {
                try {
                    completionWorkers.awaitTermination(1, TimeUnit.DAYS);
                    rejectedTaskWorker.awaitTermination(1, TimeUnit.DAYS);
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        } catch (Throwable shutdownFailure) {
            failure = Failures.append(failure, shutdownFailure);
        } finally {
            closeResult.complete(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

}
