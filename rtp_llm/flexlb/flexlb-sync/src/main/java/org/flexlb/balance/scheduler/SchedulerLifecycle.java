package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Coordinates placement shutdown and settlement of every exact request owner. */
final class SchedulerLifecycle {
    private final RequestRepository requests;
    private final AbstractRequestScheduler scheduler;

    SchedulerLifecycle(RequestRepository requests, AbstractRequestScheduler scheduler) {
        this.requests = Objects.requireNonNull(requests, "requests");
        this.scheduler = scheduler;
    }

    void closePlacement() {
        if (scheduler != null) { scheduler.closePlacement(); }
    }

    void awaitAdmissionMutations() {
        if (scheduler != null) { scheduler.awaitAdmissionMutations(); }
    }

    void abandonOutstandingDeliveries() {
        Throwable failure = null;
        for (var context : requests.snapshotActive()) {
            failure = Failures.run(failure,
                    () -> context.scheduler().cancelRequest(context, 0L, CancelReason.SHUTDOWN));
        }
        Failures.rethrow(failure, "request shutdown cancellation failed");
    }

    void awaitDeliveryCleanup() {
        var active = requests.snapshotActive();
        var pending = active.stream().map(RequestContext::delivery)
                .filter(Objects::nonNull).map(delivery -> delivery.settlement().toCompletableFuture())
                .toArray(CompletableFuture[]::new);
        try {
            CompletableFuture.allOf(pending).get(DeliveryCleanupTask.CLEANUP_TIMEOUT_MS + 1_000L,
                    TimeUnit.MILLISECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted waiting for delivery cleanup", interrupted);
        } catch (ExecutionException | TimeoutException failure) {
            throw new IllegalStateException("Unsettled deliveries at shutdown: "
                    + active.stream().map(RequestContext::getRequestId).toList(), failure);
        }
    }

    void closeOutstandingAndTerminalize() {
        Throwable failure = null;
        if (scheduler != null) {
            failure = Failures.run(failure, scheduler::closeOutstandingAndTerminalize);
        }
        var otherOwners = requests.snapshotActive().stream().map(RequestContext::scheduler)
                .filter(owner -> owner != scheduler).distinct().toList();
        for (var owner : otherOwners) {
            failure = Failures.run(failure, owner::closeOutstandingAndTerminalize);
        }
        Failures.rethrow(failure, "request shutdown finalization failed");
    }
}
