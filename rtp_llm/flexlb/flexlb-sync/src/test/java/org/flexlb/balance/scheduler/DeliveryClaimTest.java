package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DeliveryClaimTest {
    @Test @org.junit.jupiter.api.Timeout(15)
    void alreadyCompletedCancelAckSettlesOutsideTaskAndContextMonitors() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            synchronized (f.context) { assertTrue(f.context.recordDeliveryAbandonmentLocked(f.claim, CancelReason.CLIENT_CANCELLED)); }
            when(f.channel.cancel(any(), anyLong(), any(), anyLong()))
                    .thenReturn(CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.REQUEST_CLEANED));
            var task = new DeliveryCleanupTask(f.claim, f.channel, SchedulerTestSupport.cleanupExecutor(f.owner.runtime),
                    work -> f.owner.runtime.executeRequestWork(f.context, work));
            var observed = f.settlement().thenRun(() -> {
                assertFalse(Thread.holdsLock(task), "settlement dependents must not inherit the cleanup-task monitor");
                assertFalse(Thread.holdsLock(f.context));
                f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            });
            task.start();
            observed.get(5, TimeUnit.SECONDS);
            f.awaitArchive();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode).release(f.reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }
    }

    @Test void decodeProofAndTerminalEffectsReleaseEachResourceOnce() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            f.applyDecodeTerminal();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            // Concurrent status and terminal effects still release each exact resource once.
            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void terminalBeforeSenderFinishesQueuesCleanupOnlyAfterProofAndShutdownDrainsIt() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            var executor = observeWorkers(f.owner.runtime);

            f.applyDecodeTerminal();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(executor, times(1)).execute(any(Runnable.class));
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertTrue(f.prefillOwned.get(), "the sender still pins Prefill ownership");
            verify(f.prefill, never()).releaseRequest(any());

            f.owner.runtime.requests().stopAccepting();
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.owner.runtime.shutdown();

            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void rejectedWorkerSubmissionRetainsTheCleanupOwnerAndRecoversOffCaller() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            f.clearDecodeStateOwnership();
            f.recordDecodeStatus(f.decode, DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.advanceDelivery(f.claim);
            Runnable terminal = f.context.scheduler().acceptDecodeStatus(f.context, f.decode,
                    DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L), System.currentTimeMillis());
            assertNotNull(terminal);
            var executor = observeWorkers(f.owner.runtime);
            var rejected = new IllegalStateException("queue acceptance rejected");
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context), "queue submission must run outside the request monitor");
                f.owner.resumeCleanup(f.context);
                f.owner.enqueueCleanup(f.context);
                assertTrue(f.prefillOwned.get(), "a synchronous event cannot consume an unaccepted queue claim");
                throw rejected;
            }).when(executor).execute(any(Runnable.class));

            f.owner.enqueueCleanup(f.context);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.prefillOwned.get(), "accepted cleanup recovers without losing its queue claim");
            assertTrue(f.owner.requests.isCurrent(f.context), "terminal effects still own the archive gate");
            terminal.run();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void runningCleanupStillAcceptsANotificationAndPreventsEarlyShutdown() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            var executor = observeWorkers(f.owner.runtime);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                assertTrue(f.prefillOwned.compareAndSet(true, false));
                return true;
            }).when(f.prefill).releaseRequest(f.claim.item);
            var shutdownFailure = new AtomicReference<Throwable>();
            Thread shutdown = new Thread(() -> {
                try { f.owner.runtime.shutdown(); }
                catch (Throwable failure) { shutdownFailure.set(failure); }
            });
            try {
                f.applyDecodeTerminal();
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                f.owner.enqueueCleanup(f.context);
                int submissions = mockingDetails(executor).getInvocations().size();
                for (int i = 0; i < 100; i++) { f.owner.enqueueCleanup(f.context); }
                assertEquals(submissions, mockingDetails(executor).getInvocations().size(),
                        "running cleanup merges notifications without queuing more tasks");
                shutdown.start();
                RequestProtocolTestSupport.awaitCondition(() -> shutdown.getState() == Thread.State.WAITING);
                assertTrue(shutdown.isAlive());
                assertTrue(f.owner.requests.isCurrent(f.context));
            } finally {
                release.countDown();
                shutdown.join(2_000);
            }
            assertFalse(shutdown.isAlive());
            assertNull(shutdownFailure.get());
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test @org.junit.jupiter.api.Timeout(15)
    void settlementAndShutdownSeeCleanupRegisteredBeforeItsSubmission() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            f.clearDecodeStateOwnership();
            Runnable terminal = f.owner.acceptDecodeStatus(f.context, f.decode,
                    DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L), System.currentTimeMillis());
            assertNotNull(terminal);
            var delegate = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var workers = observeWorkers(f.owner.runtime);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                delegate.execute(call.getArgument(0));
                return null;
            }).when(workers).execute(any(Runnable.class));
            var producer = new Thread(() -> f.owner.enqueueCleanup(f.context));
            var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(f.owner.runtime));
            try {
                producer.start();
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                f.owner.advanceDelivery(f.claim);
                assertTrue(f.settlement().isDone(), "exact proof may settle while local cleanup is being submitted");
                terminal.run();
                closer.start();
                RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
                assertTrue(closer.isAlive(), "pre-registered cleanup blocks executor close before execute returns");
                assertTrue(f.owner.requests.isCurrent(f.context));
            } finally {
                release.countDown();
                producer.join(5_000);
                closer.join(5_000);
            }
            assertFalse(closer.isAlive());
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
        }
    }

    private static java.util.concurrent.ExecutorService observeWorkers(SchedulerRuntime runtime) {
        var original = SchedulerTestSupport.requestWorkers(runtime);
        var observed = mock(java.util.concurrent.ExecutorService.class,
                org.mockito.AdditionalAnswers.delegatesTo(original));
        ReflectionTestUtils.setField(runtime, "requestWorkers", observed);
        return observed;
    }

    @Test @org.junit.jupiter.api.Timeout(15)
    void acceptedStatusBlocksCloseBeforeItsEffectIsSubmitted() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            var delegate = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var workers = observeWorkers(f.owner.runtime);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                delegate.execute(call.getArgument(0));
                return null;
            }).when(workers).execute(any(Runnable.class));
            var source = new Thread(f::applyDecodeTerminal);
            var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(f.owner.runtime));
            try {
                source.start();
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                synchronized (f.context) { assertTrue(f.context.hasTerminalAction()); }
                closer.start();
                RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
                assertTrue(closer.isAlive(), "the terminal action has an owner before worker submission");
            } finally {
                release.countDown();
                source.join(5_000);
                closer.join(5_000);
            }
            assertFalse(closer.isAlive());
            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
        }
    }

    @Test void cancellationBeforeSendProducesNoRemoteCancel() throws Exception {
        try (var f = new Fixture()) {
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(f.owner.tryStartSend(f.claim));
            assertFalse(f.settlement().isDone(), "the sender still owns the prepared delivery");
            verifyNoInteractions(f.channel);
            f.owner.completeDelivery(f.claim, DeliveryResult.notSent(new IllegalStateException("cancelled before send")));
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
        }
    }

    @Test void earlyRemoteCleanupCannotReleaseTheSenderObligation() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            verify(f.channel).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), anyLong());
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            SchedulerTestSupport.runtime(f.owner).requests().stopAccepting();

            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
            assertFalse(f.context.getFuture().join().isSuccess(), "late ACK cannot replace cancellation");
            SchedulerTestSupport.runtime(f.owner).shutdown();

        }
    }

    @Test void prefillFenceWaitsForExactDecodeSettlement() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.onResponseUndeliverable(f.context);
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
            assertFalse(f.settlement().isDone());
            f.recordDecodeStatus(RequestProtocolTestSupport.decodeEndpoint(), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.advanceDelivery(f.claim);
            assertFalse(f.settlement().isDone(), "another endpoint is not proof");
            f.recordDecodeStatus(f.decode, DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L));
            f.owner.advanceDelivery(f.claim);
            assertFalse(f.settlement().isDone(), "another reservation is not proof");
            f.applyDecodeTerminal();
            f.settlement().get(2, TimeUnit.SECONDS);
        }
    }

    @Test void failedResponseDeliveryAfterSelectedSuccessStillCancelsEngine() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            assertTrue(f.context.getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.owner.onResponseUndeliverable(f.context);
            verify(f.channel).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), anyLong());
            assertTrue(f.context.getFuture().join().isSuccess(), "selected result remains immutable");
            assertFalse(f.settlement().isDone());
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
        }
    }

    @Test void rejectedCleanupExecutorRecordsFailureWithoutArchiving() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            SchedulerTestSupport.cleanupExecutor(f.owner.runtime).shutdown();
            f.owner.cancel(41L, 0L, CancelReason.SHUTDOWN);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            assertFalse(f.settlement().isDone(), "executor failure is not remote cleanup proof");
            assertTrue(f.owner.requests.isCurrent(f.context));
            org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void cleanupTimeoutWithoutProofKeepsTheResourceObligationPending() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());

            timeout.run();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            assertAll(
                    () -> assertFalse(f.settlement().isDone(), "cleanup execution failure is not resource settlement"),
                    () -> assertTrue(f.owner.requests.isCurrent(f.context)),
                    () -> assertNull(f.context.deliveryReleaseReason(f.claim)),
                    () -> verify(f.prefill, never()).releaseRequest(any()),
                    () -> verify(f.decode, never()).release(any(), any()));
        }
    }

    @Test void activeDecodeHeartbeatCannotPublishSuccessWhileNotSentEffectsArePending() throws Exception {
        try (var f = new Fixture()) {
            var original = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var held = mock(java.util.concurrent.ExecutorService.class);
            var accepted = new java.util.ArrayList<Runnable>();
            org.mockito.Mockito.doAnswer(call -> {
                accepted.add(call.getArgument(0));
                return null;
            }).when(held).execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", held);
            try {
                f.owner.completeDelivery(f.claim, DeliveryResult.notSent(new IllegalStateException("not sent")));
                assertFalse(accepted.isEmpty(), "failure effects are deliberately paused");
                Runnable heartbeat = f.owner.acceptDecodeStatus(f.context, f.decode,
                        DecodeResources.DecodeRequestStatus.active(f.reservation), System.currentTimeMillis());
                if (heartbeat != null) { heartbeat.run(); }
                assertFalse(f.context.getFuture().isDone(), "ACTIVE cannot replace the selected send failure");
                assertTrue(f.settlement().isDone(), "NOT_SENT itself proves delivery settlement");
            } finally {
                ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", original);
                for (Runnable work : accepted) { work.run(); }
            }
            f.awaitArchive();
            assertFalse(f.context.getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test void notSentQueuesCleanupBeforePublishingTheSettlementBarrier() throws Exception {
        try (var f = new Fixture()) {
            var original = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var held = mock(java.util.concurrent.ExecutorService.class);
            var accepted = new java.util.ArrayList<Runnable>();
            org.mockito.Mockito.doAnswer(call -> {
                assertFalse(f.settlement().isDone(), "cleanup must be accepted before the settlement barrier opens");
                accepted.add(call.getArgument(0));
                return null;
            }).when(held).execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", held);
            try {
                f.owner.completeDelivery(f.claim, DeliveryResult.notSent(new IllegalStateException("not sent")));
                assertFalse(accepted.isEmpty(), "the failure must have a cleanup owner");
                assertTrue(f.settlement().isDone(), "NOT_SENT proves remote resources can be released");
            } finally {
                ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", original);
                for (Runnable work : accepted) { work.run(); }
            }
            f.awaitArchive();
            assertFalse(f.context.getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertNull(f.context.deliveryCleanupEvidence(f.claim).reason(), "NOT_SENT needs no remote abandonment protocol");
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.decode).release(f.reservation, DecodeResources.ReleaseReason.NOT_SENT);
            verify(f.channel, never()).cancel(any(), anyLong(), any(), anyLong());
        }
    }

    @Test void lateCleanedAckAfterTimeoutArchivesOnlyTheOriginalRequest() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            Response selected = f.context.getFuture().get(2, TimeUnit.SECONDS);

            timeout.run();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertSame(selected, f.context.getFuture().join(), "cleanup cannot replace the selected response");

            var terminal = f.owner.requests.findTerminal(41L);
            assertTrue(f.owner.requests.removeExactTerminal(terminal, Long.MAX_VALUE));
            var replacement = RequestProtocolTestSupport.context(f.context.getConfig(), 41L);
            var replacementFuture = RequestProtocolTestSupport.register(f.owner, replacement);
            try {
                f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
                f.applyDecodeTerminal();
                f.owner.onPrefillGenerationRetired(f.prefill, f.claim.item);
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

                assertSame(replacement, f.owner.requests.findActive(41L));
                assertFalse(replacementFuture.isDone(), "old cleanup facts cannot finish a reused request identity");
                verify(f.prefill, times(1)).releaseRequest(f.claim.item);
                f.assertDecodeOwnershipReleasedOnce();
            } finally {
                replacementFuture.completeExceptionally(new IllegalStateException("test cleanup"));
            }
        }
    }

    @Test void endpointRetirementAfterCleanupTimeoutStillArchivesTheRequest() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());

            timeout.run();
            f.owner.onPrefillGenerationRetired(f.prefill, f.claim.item);
            f.retireDecode();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            assertTrue(f.context.deliveryCleanupEvidence(f.claim).remoteSettled(), "both exact generations have retired");
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void lateFenceAfterTimeoutWaitsForTheExactDecodeTerminal() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());

            timeout.run();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
            f.owner.onDecodeStatus(f.context, RequestProtocolTestSupport.decodeEndpoint(), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.onDecodeStatus(f.context, f.decode, DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone(), "Prefill fencing and foreign Decode facts do not settle the request");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.decode, never()).release(any(), any());

            f.applyDecodeTerminal();
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void lateCleanedAckAfterTimeoutStillWaitsForSenderExit() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);

            timeout.run();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone(), "remote cleanup cannot settle an active sender");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());

            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class,
            names = {"ACCEPTED", "NOT_FOUND", "UNSUPPORTED", "FAILED"})
    void acknowledgementWithoutCleanupProofCannotReleaseAnUncertainSend(EngineCancelChannel.CancelAck ack) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.onResponseUndeliverable(f.context);
            f.cancel.complete(ack);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.owner.tryStartSend(f.claim), "uncertain delivery must not be sent again");
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertNull(f.context.deliveryReleaseReason(f.claim));
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());

            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void fenceDecodeProofAndSenderExitSettleInAnyOrder(boolean decodeFirst, boolean senderFirst) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            if (senderFirst) { f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost"))); }
            Runnable decodeProof = f::applyDecodeTerminal;
            if (decodeFirst) { decodeProof.run(); }
            else { f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED); }
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone(), "one side of the cleanup proof cannot settle delivery");
            verify(f.prefill, never()).releaseRequest(any());

            if (decodeFirst) { f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED); }
            else { decodeProof.run(); }
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            if (!senderFirst) {
                assertFalse(f.settlement().isDone(), "complete remote proof still needs sender exit");
                f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            }
            f.assertArchivedAfterCleanup();
            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            decodeProof.run();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            f.assertDecodeOwnershipReleasedOnce();
        }
    }

    @Test void repeatedCancellationPreservesFirstReasonAndStartsOneCleanup() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            assertFalse(f.owner.tryStartSend(f.claim), "send qualification is consumed once");
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.cancel(41L, 0L, CancelReason.SHUTDOWN);
            f.owner.onResponseUndeliverable(f.context);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.channel, times(1)).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), anyLong());
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("cleanupEventOrders")
    void everyCleanupEventPermutationRequiresAllProofAndReleasesOnce(String outcome, String order) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            for (int index = 0; index < order.length(); index++) {
                switch (order.charAt(index)) {
                    case 'S' -> f.owner.completeDelivery(f.claim, switch (outcome) {
                        case "DELIVERED" -> DeliveryResult.delivered();
                        case "REJECTED" -> DeliveryResult.prefillRejected(new IllegalStateException("rejected"));
                        case "UNKNOWN" -> DeliveryResult.uncertain(new IllegalStateException("reply lost"));
                        default -> throw new AssertionError(outcome);
                    });
                    case 'P' -> f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
                    case 'D' -> f.applyDecodeTerminal();
                    default -> throw new AssertionError(order);
                }
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
                if (index < order.length() - 1) {
                    assertFalse(f.settlement().isDone(), "partial evidence: " + order.substring(0, index + 1));
                    assertTrue(f.owner.requests.isCurrent(f.context));
                    verify(f.prefill, never()).releaseRequest(any());
                }
            }
            f.assertArchivedAfterCleanup();
            assertFalse(f.owner.tryStartSend(f.claim));
            assertThrows(IllegalStateException.class, () -> f.owner.completeDelivery(f.claim, DeliveryResult.delivered()));
            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.applyDecodeTerminal();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            f.assertDecodeOwnershipReleasedOnce();
        }
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> cleanupEventOrders() {
        return java.util.stream.Stream.of("DELIVERED", "REJECTED", "UNKNOWN").flatMap(outcome ->
                java.util.stream.Stream.of("SPD", "SDP", "PSD", "PDS", "DSP", "DPS")
                        .map(order -> org.junit.jupiter.params.provider.Arguments.of(outcome, order)));
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"foreignEndpoint", "foreignToken", "foreignGeneration", "foreignRequest", "notTerminal"})
    void foreignDecodeEvidenceCannotSettleTheExactDelivery(String mismatch) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
            DecodeEndpoint source = mismatch.equals("foreignEndpoint") ? RequestProtocolTestSupport.decodeEndpoint() : f.decode;
            var reservation = switch (mismatch) {
                case "foreignToken" -> new DecodeResources.ReservationHandle(1L, 41L, 2L);
                case "foreignGeneration" -> new DecodeResources.ReservationHandle(2L, 41L, 1L);
                case "foreignRequest" -> new DecodeResources.ReservationHandle(1L, 42L, 1L);
                default -> f.reservation;
            };
            var requestStatus = mismatch.equals("notTerminal")
                    ? DecodeResources.DecodeRequestStatus.active(reservation)
                    : DecodeResources.DecodeRequestStatus.terminal(reservation, 0L);
            f.recordDecodeStatus(source, requestStatus);
            f.owner.advanceDelivery(f.claim);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone());
            assertNull(f.context.deliveryReleaseReason(f.claim));
            assertTrue(f.decodeOwned.get(), "foreign facts cannot release the exact Decode owner");
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());
            f.applyDecodeTerminal();
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8})
    void cleanupCanSucceedAtEveryBoundedRetryAndCancelsItsDeadline(int successfulAttempt) throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var replies = new java.util.concurrent.CopyOnWriteArrayList<CompletableFuture<EngineCancelChannel.CancelAck>>();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                var reply = new CompletableFuture<EngineCancelChannel.CancelAck>();
                replies.add(reply);
                return reply;
            });
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            for (int attempt = 1; attempt <= successfulAttempt; attempt++) {
                assertEquals(attempt, replies.size());
                replies.get(attempt - 1).complete(attempt == successfulAttempt
                        ? EngineCancelChannel.CancelAck.REQUEST_CLEANED : EngineCancelChannel.CancelAck.NOT_FOUND);
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
                if (attempt < successfulAttempt) {
                    assertFalse(f.settlement().isDone());
                    verify(f.prefill, never()).releaseRequest(any());
                    timer.runRetry(Math.min(400L, 25L << (attempt - 1)));
                }
            }
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(successfulAttempt)).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), eq(500L));
            verify(timer.deadline).cancel(false);
            assertTrue(timer.retries.isEmpty());
        }
    }

    @Test void timedOutAcknowledgementStillSuppliesLateCleanupProof() throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var replies = new java.util.concurrent.CopyOnWriteArrayList<CompletableFuture<EngineCancelChannel.CancelAck>>();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> {
                var reply = new CompletableFuture<EngineCancelChannel.CancelAck>();
                replies.add(reply);
                return reply;
            });
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertEquals(1, replies.size());

            timer.runAckTimeout();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone());
            timer.runRetry(25L);
            assertEquals(2, replies.size());

            replies.getFirst().complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(2)).cancel(any(), eq(41L), any(), eq(500L));
            assertTrue(timer.retries.isEmpty());
        }
    }

    @Test void overallTimeoutStopsAnAlreadyScheduledRetry() throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenReturn(
                    CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.NOT_FOUND));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.channel, times(1)).cancel(any(), eq(41L), any(), eq(500L));

            timer.runDeadline();
            timer.runRetry(25L); // A callback already taken from the timer queue may still run.
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.channel, times(1)).cancel(any(), eq(41L), any(), eq(500L));
            assertFalse(f.settlement().isDone());
            org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void retryExhaustionStillAcceptsAnEarlierTimedOutAttemptProof() throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var first = new CompletableFuture<EngineCancelChannel.CancelAck>();
            var calls = new java.util.concurrent.atomic.AtomicInteger();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call ->
                    calls.incrementAndGet() == 1 ? first
                            : CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.NOT_FOUND));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            timer.runAckTimeout();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            for (int attempt = 1; attempt < 8; attempt++) {
                timer.runRetry(Math.min(400L, 25L << (attempt - 1)));
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            }
            assertEquals(8, calls.get());
            org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(f.owner));
            assertFalse(f.settlement().isDone());
            assertTrue(timer.retries.isEmpty());

            first.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertEquals(8, calls.get(), "late proof cannot restart exhausted retries");
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class,
            names = {"ACCEPTED", "NOT_FOUND", "FAILED"})
    void retryExhaustionCannotInventResourceRelease(EngineCancelChannel.CancelAck ack) throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> CompletableFuture.completedFuture(ack));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            for (int attempt = 1; attempt < 8; attempt++) {
                verify(f.channel, times(attempt)).cancel(any(), eq(41L), any(), anyLong());
                timer.runRetry(Math.min(400L, 25L << (attempt - 1)));
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            }
            verify(f.channel, times(8)).cancel(any(), eq(41L), any(), anyLong());
            assertTrue(timer.retries.isEmpty(), "retry budget must end without scheduling a ninth attempt");
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertNull(f.context.deliveryReleaseReason(f.claim));
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void unsupportedCancelStopsRetriesWithoutChangingCleanupTimeout(boolean timeout) throws Exception {
        try (var logs = new CleanupLogs(); var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.cancel.complete(EngineCancelChannel.CancelAck.UNSUPPORTED);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            assertTrue(timer.retries.isEmpty(), "unsupported cancellation must not schedule another attempt");
            verify(timer.deadline, never()).cancel(anyBoolean());
            assertEquals(1, logs.count(ch.qos.logback.classic.Level.WARN));
            assertNull(SchedulerTestSupport.failure(f.owner));
            assertFalse(f.settlement().isDone());
            assertTrue(f.prefillOwned.get());
            assertTrue(f.decodeOwned.get());

            if (timeout) {
                timer.runDeadline();
                timer.runDeadline();
                assertNotNull(SchedulerTestSupport.failure(f.owner), "the existing overall timeout still records failure");
                assertFalse(f.settlement().isDone());
                assertTrue(f.prefillOwned.get());
                assertTrue(f.decodeOwned.get());
            }
            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            timer.runDeadline(); // A canceled callback already taken from the timer queue is harmless.
            verify(f.channel, times(1)).cancel(any(), eq(41L), any(), anyLong());
            if (timeout) { assertThrows(IllegalStateException.class, f.owner.runtime::shutdown); }
            else { assertDoesNotThrow(f.owner.runtime::shutdown); }
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void lateUnsupportedStopsQueuedOrInFlightRetryAndStillAcceptsProof(boolean retryStarted) throws Exception {
        try (var logs = new CleanupLogs(); var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var replies = new java.util.ArrayList<CompletableFuture<EngineCancelChannel.CancelAck>>();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> {
                var reply = new CompletableFuture<EngineCancelChannel.CancelAck>();
                replies.add(reply);
                return reply;
            });
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            timer.runAckTimeout();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            if (retryStarted) { timer.runRetry(25L); }

            replies.getFirst().complete(EngineCancelChannel.CancelAck.UNSUPPORTED);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            if (!retryStarted) { timer.runRetry(25L); }
            else { timer.runAckTimeout(); }
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertEquals(retryStarted ? 2 : 1, replies.size());
            assertTrue(timer.retries.isEmpty(), "no new retries after late unsupported evidence");
            assertEquals(1, logs.count(ch.qos.logback.classic.Level.WARN));
            verify(timer.deadline, never()).cancel(anyBoolean());
            assertNull(SchedulerTestSupport.failure(f.owner));
            assertFalse(f.settlement().isDone());

            if (retryStarted) {
                replies.getLast().complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            } else {
                f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            }
            f.assertArchivedAfterCleanup();
            assertNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class, names = {"NOT_FOUND", "ACCEPTED", "FAILED"})
    void lateReplyWithoutProofCannotScheduleDuplicateRetry(EngineCancelChannel.CancelAck ack) throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            timer.runAckTimeout();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertEquals(1, timer.retries.size());

            f.cancel.complete(ack);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertEquals(1, timer.retries.size(), "timeout already scheduled this attempt's retry");
            assertFalse(f.settlement().isDone());
            assertNull(SchedulerTestSupport.failure(f.owner));
            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
        }
    }

    private static final class CleanupLogs implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        private final ch.qos.logback.classic.Level previousLevel = logger.getLevel();
        private final ch.qos.logback.core.read.ListAppender<ch.qos.logback.classic.spi.ILoggingEvent> appender =
                new ch.qos.logback.core.read.ListAppender<>();

        CleanupLogs() {
            logger.setLevel(ch.qos.logback.classic.Level.WARN);
            appender.start();
            logger.addAppender(appender);
        }

        long count(ch.qos.logback.classic.Level level) {
            return appender.list.stream().filter(event -> event.getLevel() == level
                    && event.getFormattedMessage().contains("[delivery-cleanup]")).count();
        }

        public void close() {
            logger.detachAppender(appender);
            appender.stop();
            logger.setLevel(previousLevel);
        }
    }

    private static final class ControlledCleanup {
        final java.util.ArrayDeque<java.util.Map.Entry<Long, Runnable>> retries = new java.util.ArrayDeque<>();
        final java.util.ArrayDeque<Runnable> acknowledgements = new java.util.ArrayDeque<>();
        final ScheduledFuture<?> deadline = mock(ScheduledFuture.class);
        Runnable deadlineAction;
        final ScheduledThreadPoolExecutor executor = spy(new ScheduledThreadPoolExecutor(1));

        ControlledCleanup() {
            doAnswer(call -> {
                long delay = call.getArgument(1);
                if (delay == DeliveryCleanupTask.CLEANUP_TIMEOUT_MS) {
                    deadlineAction = call.getArgument(0);
                    return deadline;
                }
                if (delay < 500L) { retries.addLast(java.util.Map.entry(delay, call.getArgument(0))); }
                if (delay == 500L) { acknowledgements.addLast(call.getArgument(0)); }
                return mock(ScheduledFuture.class);
            }).when(executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
        }

        void runRetry(long expectedDelay) {
            var retry = retries.pollFirst();
            assertNotNull(retry, "cleanup must schedule a retry");
            assertEquals(expectedDelay, retry.getKey().longValue());
            retry.getValue().run();
        }

        void runAckTimeout() {
            Runnable timeout = acknowledgements.pollFirst();
            assertNotNull(timeout, "cancel attempt must have an acknowledgement deadline");
            timeout.run();
        }

        void runDeadline() {
            assertNotNull(deadlineAction, "cleanup must have an overall deadline");
            deadlineAction.run();
        }
    }

    private static final class Fixture implements AutoCloseable {
        final EngineCancelChannel channel = mock(EngineCancelChannel.class);
        final CompletableFuture<EngineCancelChannel.CancelAck> cancel = new CompletableFuture<>();
        final AbstractRequestScheduler owner;
        final RequestContext context;
        final PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        final java.util.concurrent.atomic.AtomicBoolean prefillOwned = new java.util.concurrent.atomic.AtomicBoolean(true);
        final java.util.concurrent.atomic.AtomicBoolean decodeOwned = new java.util.concurrent.atomic.AtomicBoolean(true);
        final java.util.concurrent.atomic.AtomicBoolean decodeClearedByState = new java.util.concurrent.atomic.AtomicBoolean();
        final DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        final DecodeResources.ReservationHandle reservation = new DecodeResources.ReservationHandle(1L, 41L, 1L);
        final RequestContext.DeliveryClaim claim;
        Fixture() {
            var config = SchedulingTestConfig.batchConfig();
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            owner = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
            ReflectionTestUtils.setField(owner.runtime, "cancelChannel", channel);
            when(channel.cancel(any(), anyLong(), any(), anyLong())).thenReturn(cancel);
            context = RequestProtocolTestSupport.context(config, 41L);
            var future = RequestProtocolTestSupport.register(owner, context);
            var response = new Response();
            response.setSuccess(true);
            var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, response, null, null,
                    prefill, decode, reservation, System.currentTimeMillis());
            when(prefill.releaseRequest(any())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(context), "resource cleanup must run outside the request monitor");
                assertSame(item, call.getArgument(0), "only this exact Prefill owner may be released");
                assertTrue(prefillOwned.compareAndSet(true, false), "Prefill ownership may be released only once");
                return true;
            });
            when(decode.release(any(), any())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(context), "Decode cleanup must run outside the request monitor");
                assertEquals(reservation, call.getArgument(0), "only this exact Decode owner may be released");
                assertTrue(decodeOwned.compareAndSet(true, false), "Decode ownership may be released only once");
                return DecodeResources.ReservationReleaseResult.RELEASED;
            });
            RequestProtocolTestSupport.bindRoute(owner, new RequestProtocolTestSupport.Registered(item, future));
            claim = RequestProtocolTestSupport.claimBatch(owner, item, 51L, () -> true);
            assertNotNull(claim);
        }
        CompletableFuture<Void> settlement() { return claim.settlement().toCompletableFuture(); }
        void awaitArchive() throws InterruptedException { RequestProtocolTestSupport.awaitCondition(() -> !owner.requests.isCurrent(context)); }

        // DecodeState removes exact ownership before Endpoint publishes a terminal or retirement fact.
        private void clearDecodeStateOwnership() {
            if (decodeOwned.getAndSet(false)) { decodeClearedByState.set(true); }
        }

        void recordDecodeStatus(DecodeEndpoint source, DecodeResources.DecodeRequestStatus status) {
            synchronized (context) { context.consumeDecodeStatusLocked(source, status, System.currentTimeMillis()); }
        }

        void applyDecodeTerminal() {
            clearDecodeStateOwnership();
            owner.onDecodeStatus(context, decode, DecodeResources.DecodeRequestStatus.terminal(reservation, 0L));
        }

        void retireDecode() {
            clearDecodeStateOwnership();
            when(decode.isRetired()).thenReturn(true);
            owner.onDecodeGenerationRetired(context, decode, reservation);
        }

        void assertDecodeOwnershipReleasedOnce() {
            assertFalse(decodeOwned.get(), "archival requires physical Decode ownership to be released");
            verify(decode, times(decodeClearedByState.get() ? 0 : 1))
                    .release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }

        ControlledCleanup controlCleanup() {
            var controlled = new ControlledCleanup();
            SchedulerTestSupport.cleanupExecutor(owner.runtime).shutdown();
            ReflectionTestUtils.setField(owner.runtime, "cleanupExecutor", controlled.executor);
            return controlled;
        }

        Runnable controlCleanupTimeout() {
            AtomicReference<Runnable> deadline = new AtomicReference<>();
            var timer = spy(new ScheduledThreadPoolExecutor(1));
            // Capture real scheduled callbacks; no wall-clock timer runs in these regressions.
            doAnswer(call -> {
                if (call.<Long>getArgument(1) == DeliveryCleanupTask.CLEANUP_TIMEOUT_MS) {
                    deadline.set(call.getArgument(0));
                }
                return mock(ScheduledFuture.class);
            }).when(timer).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            SchedulerTestSupport.cleanupExecutor(owner.runtime).shutdown();
            ReflectionTestUtils.setField(owner.runtime, "cleanupExecutor", timer);
            return () -> {
                assertNotNull(deadline.get(), "the cleanup task must install its overall deadline");
                deadline.get().run();
            };
        }

        void assertArchivedAfterCleanup() throws Exception {
            settlement().get(2, TimeUnit.SECONDS);
            synchronized (context) {
                assertEquals(CancelReason.CLIENT_CANCELLED, ReflectionTestUtils.getField(ReflectionTestUtils.getField(context, "deliveryState"), "abandonmentReason"));
                assertEquals(true, ReflectionTestUtils.getField(ReflectionTestUtils.getField(context, "deliveryState"), "prefillReleaseProven"));
                assertEquals(true, ReflectionTestUtils.getField(ReflectionTestUtils.getField(context, "deliveryState"), "decodeReleaseProven"));
            }
            awaitArchive();
            SchedulerTestSupport.awaitRequestWork(owner.runtime);
            assertFalse(context.getFuture().join().isSuccess());
            assertFalse(prefillOwned.get(), "archival requires Prefill ownership to be released");
            verify(prefill, times(1)).releaseRequest(claim.item);
            assertDecodeOwnershipReleasedOnce();
        }
        public void close() {
            SchedulerTestSupport.runtime(owner).requests().stopAccepting();
            ((QueuedRequestScheduler)owner).close();
            SchedulerTestSupport.timer(owner.runtime).close();
            SchedulerTestSupport.closeRequestExecutors(owner.runtime);
        }
    }
}
