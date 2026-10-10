package org.flexlb.balance.scheduler;

/** Request work capabilities without access to shared executor lifecycle. */
public interface RequestTaskScheduler {
    @FunctionalInterface
    interface Cancellation {
        void cancel();
    }

    /** Register a one-shot timeout; the handle can only cancel this task. */
    Cancellation scheduleTimeout(Runnable work, long delayMs);

    /** Serialize work with the other continuations of this exact request. */
    void executeContinuation(RequestContext context, Runnable work);
}
