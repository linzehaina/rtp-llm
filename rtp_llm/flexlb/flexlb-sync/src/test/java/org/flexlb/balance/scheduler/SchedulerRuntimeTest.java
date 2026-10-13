package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigTestSupport;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class SchedulerRuntimeTest {
    @Test
    @org.junit.jupiter.api.Timeout(15)
    void externalFutureCancellationRemainsSynchronousAndItsSourceBlocksWorkClose() throws Exception {
        try (var f = new Fixture(false)) {
            var scheduler = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            var context = RequestProtocolTestSupport.context(f.config, 9103L);
            var result = RequestProtocolTestSupport.register(scheduler, context);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            var cancelled = new java.util.concurrent.CompletableFuture<Boolean>();
            var callback = result.whenComplete((response, failure) -> {
                assertFalse(Thread.holdsLock(context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
            });
            var source = new Thread(() -> {
                try { cancelled.complete(result.cancel(false)); }
                catch (Throwable failure) { cancelled.completeExceptionally(failure); }
            });
            var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(f.runtime));
            try {
                source.start();
                assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS));
                assertTrue(result.isCancelled(), "Future.cancel selects and publishes its result on the caller");
                assertFalse(cancelled.isDone(), "synchronous dependent callbacks still belong to that caller");
                closer.start();
                RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
                assertTrue(closer.isAlive(), "drain includes the registered cancellation source");
            } finally {
                release.countDown();
                source.join(5_000);
                closer.join(5_000);
            }
            assertTrue(cancelled.get(5, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue(callback.isCompletedExceptionally());
            assertFalse(closer.isAlive());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void directStartupUsesOneSchedulerAndSharedConfiguration() {
        try (var f = new Fixture(false)) {
            var scheduler = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            assertInstanceOf(DirectRequestScheduler.class, scheduler);
            assertSame(scheduler, SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(f.config, scheduler.config);
            assertSame(f.config, new RequestContext(f.config).getConfig());
            assertThrows(IllegalStateException.class,
                    () -> f.runtime.registerScheduler(mock(AbstractRequestScheduler.class)));
        }
    }

    @Test
    void queueStartupUsesOneScheduler() {
        try (var f = new Fixture(true)) {
            assertInstanceOf(QueuedRequestScheduler.class, SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(SchedulerTestSupport.initializedScheduler(f.runtime), SchedulerTestSupport.initializedScheduler(f.runtime));
            assertThrows(IllegalStateException.class,
                    () -> f.runtime.registerScheduler(mock(AbstractRequestScheduler.class)));
        }
    }

    @Test
    void schedulerCannotBeRegisteredAfterShutdown() {
        var f = new Fixture(false);
        f.close();
        assertThrows(IllegalStateException.class,
                () -> f.runtime.registerScheduler(mock(AbstractRequestScheduler.class)));
    }

    @Test
    void shutdownInvokesSchedulerHooksAndContinuesAfterFailure() {
        var config = SchedulingTestConfig.newConfig();
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var endpoints = mock(EndpointRegistry.class);
        var dispatcher = mock(DefaultBatchDispatcher.class);
        var runtime = new SchedulerRuntime(new RequestRepository(), endpoints,
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), dispatcher,
                service, mock(EngineCancelChannel.class));
        var scheduler = mock(AbstractRequestScheduler.class);
        var expected = new IllegalStateException("placement close failed");
        doThrow(expected).when(scheduler).closePlacement();
        runtime.registerScheduler(scheduler);

        assertSame(expected, assertThrows(IllegalStateException.class, runtime::shutdown));

        var order = inOrder(scheduler, dispatcher, endpoints);
        order.verify(scheduler).closePlacement();
        order.verify(scheduler).awaitAdmissionMutations();
        order.verify(dispatcher).shutdownAndAwait();
        order.verify(endpoints).close();
        order.verify(scheduler).closeOutstandingAndTerminalize();
    }

    @Test
    void stoppingRejectsRegistrationWhileAcceptedRequestWorkCanContinue() throws Exception {
        try (var f = new Fixture()) {
            var scheduler = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            var accepted = RequestProtocolTestSupport.context(f.config, 9101L);
            var result = RequestProtocolTestSupport.register(scheduler, accepted);
            assertFalse(result.isDone());

            f.runtime.requests().stopAccepting();

            assertFalse(f.runtime.isAccepting());
            assertFalse(f.runtime.requests().isClosed());
            assertSame(accepted, f.runtime.requests().findActive(accepted.getRequestId()));
            var rejected = RequestProtocolTestSupport.context(f.config, 9102L);
            var rejectedResult = RequestProtocolTestSupport.register(scheduler, rejected);
            assertFalse(rejectedResult.join().isSuccess());
            assertNull(rejected.scheduler());

            var continued = new java.util.concurrent.CompletableFuture<Void>();
            f.runtime.executeRequestWork(accepted, () -> continued.complete(null));
            continued.get(1, java.util.concurrent.TimeUnit.SECONDS);
            assertFalse(result.isDone());
        }
    }

    @Test
    void cancellingOneTimeoutLeavesOtherTimeoutsAvailable() throws Exception {
        try (var f = new Fixture()) {
            var cancelledWork = new java.util.concurrent.CompletableFuture<Void>();
            var cancellation = f.runtime.scheduleTimeout(() -> cancelledWork.complete(null), 60_000L);
            cancellation.cancel();

            var otherWork = new java.util.concurrent.CompletableFuture<Void>();
            f.runtime.scheduleTimeout(() -> otherWork.complete(null), 0L);
            otherWork.get(1, java.util.concurrent.TimeUnit.SECONDS);
            assertFalse(cancelledWork.isDone());
            assertTrue(SchedulerTestSupport.cleanupExecutor(f.runtime).getQueue().isEmpty());
        }
    }

    @Test
    void shutdownCancelsEveryExactOwnerInTheSharedRequestDirectory() {
        try (var f = new Fixture()) {
            var installed = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            var other = new DirectRequestScheduler(mock(RequestWorkerSelector.class), f.runtime, f.config,
                    SchedulingTestConfig.capacity(f.config), mock(RecentCacheKeyTraceReporter.class));
            var installedResult = RequestProtocolTestSupport.register(installed,
                    RequestProtocolTestSupport.context(f.config, 9103L));
            var otherResult = RequestProtocolTestSupport.register(other,
                    RequestProtocolTestSupport.context(f.config, 9104L));

            f.runtime.shutdown();

            assertFalse(installedResult.join().isSuccess());
            assertFalse(otherResult.join().isSuccess());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void failedCancellationDoesNotPreventOtherOwnersFromSettling() {
        try (var f = new Fixture()) {
            var installed = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            var other = spy(new DirectRequestScheduler(mock(RequestWorkerSelector.class), f.runtime, f.config,
                    SchedulingTestConfig.capacity(f.config), mock(RecentCacheKeyTraceReporter.class)));
            var otherContext = RequestProtocolTestSupport.context(f.config, 9105L);
            var otherResult = RequestProtocolTestSupport.register(other, otherContext);
            var installedResult = RequestProtocolTestSupport.register(installed,
                    RequestProtocolTestSupport.context(f.config, 9106L));
            var expected = new IllegalStateException("cancellation failed");
            doThrow(expected).when(other).cancelRequest(otherContext, 0L, CancelReason.SHUTDOWN);

            assertSame(expected, assertThrows(IllegalStateException.class, f.runtime::shutdown));

            assertFalse(installedResult.join().isSuccess());
            assertFalse(otherResult.join().isSuccess());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void shutdownFinalizesAnUnregisteredOwnerAfterCancellationFailure() {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var endpoints = mock(EndpointRegistry.class);
        var runtime = new SchedulerRuntime(new RequestRepository(), endpoints, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(DefaultBatchDispatcher.class), service, mock(EngineCancelChannel.class));
        try {
            var owner = spy(new DirectRequestScheduler(mock(RequestWorkerSelector.class), runtime, config,
                    SchedulingTestConfig.capacity(config), mock(RecentCacheKeyTraceReporter.class)));
            var context = RequestProtocolTestSupport.context(config, 9107L);
            var result = RequestProtocolTestSupport.register(owner, context);
            var expected = new IllegalStateException("cancellation failed before scheduler registration");
            doThrow(expected).when(owner).cancelRequest(context, 0L, CancelReason.SHUTDOWN);

            assertSame(expected, assertThrows(IllegalStateException.class, runtime::shutdown));

            assertFalse(result.join().isSuccess());
            assertEquals(0, runtime.requests().liveRequestCount());
            verify(endpoints).close();
            assertTrue(SchedulerTestSupport.cleanupExecutor(runtime).isTerminated());
        } finally {
            runtime.shutdown();
        }
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final EndpointRegistry endpoints = mock(EndpointRegistry.class);
        final SchedulerRuntime runtime;
        Fixture() { this(false); }
        Fixture(boolean queued) {
            config.setScheduler(queued ? new SchedulerConfig() : ConfigTestSupport.direct());
            config.setDispatcher(ConfigTestSupport.nonBatch());
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            runtime = new SchedulerRuntime(new RequestRepository(), endpoints, reporter,
                    mock(RequestSchedulerReporter.class), mock(DefaultBatchDispatcher.class), service, mock(EngineCancelChannel.class));
            runtime.registerScheduler(PlacementConfiguration.create(runtime, config,
                    mock(RequestWorkerSelector.class), reporter, mock(DecodeCapacityAcquirer.class), new PlacementAvailability(), SchedulingTestConfig.capacity(config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class)));
        }
        public void close() { runtime.shutdown(); }
    }
}
