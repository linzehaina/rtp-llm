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
    void directStartupUsesOneSchedulerAndSharedConfiguration() {
        try (var f = new Fixture(false)) {
            var scheduler = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(f.runtime);
            assertInstanceOf(DirectRequestScheduler.class, scheduler);
            assertSame(scheduler, SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(f.config, scheduler.config);
            assertSame(f.config, new RequestContext(f.config).getConfig());
            assertThrows(IllegalStateException.class,
                    () -> f.runtime.initializeScheduler(mock(SchedulerLifecycle.class)));
        }
    }

    @Test
    void queueStartupUsesOneScheduler() {
        try (var f = new Fixture(true)) {
            assertInstanceOf(QueuedRequestScheduler.class, SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(SchedulerTestSupport.initializedScheduler(f.runtime), SchedulerTestSupport.initializedScheduler(f.runtime));
            assertThrows(IllegalStateException.class,
                    () -> f.runtime.initializeScheduler(mock(SchedulerLifecycle.class)));
        }
    }

    @Test
    void stoppedRuntimeCannotBeInitializedAgain() {
        var f = new Fixture(false);
        f.close();
        assertThrows(IllegalStateException.class,
                () -> f.runtime.initializeScheduler(mock(SchedulerLifecycle.class)));
    }

    @Test
    void shutdownUsesTheSchedulerLifecycleHookAndContinuesAfterItsFailure() {
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
        runtime.initializeScheduler(new SchedulerLifecycle(runtime.requests(), scheduler));

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

            f.runtime.stopAccepting();

            assertFalse(f.runtime.isAccepting());
            assertFalse(f.runtime.requests().isClosed());
            assertSame(accepted, f.runtime.requests().findActive(accepted.getRequestId()));
            var rejected = RequestProtocolTestSupport.context(f.config, 9102L);
            var rejectedResult = RequestProtocolTestSupport.register(scheduler, rejected);
            assertFalse(rejectedResult.join().isSuccess());
            assertNull(rejected.scheduler());

            var continued = new java.util.concurrent.CompletableFuture<Void>();
            f.runtime.executeContinuation(accepted, () -> continued.complete(null));
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
            runtime.initializeScheduler(PlacementConfiguration.create(runtime, config,
                    mock(RequestWorkerSelector.class), reporter, mock(DecodeCapacityAcquirer.class), new PlacementAvailability(), SchedulingTestConfig.capacity(config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class)).shutdownLifecycle());
        }
        public void close() { runtime.shutdown(); }
    }
}
