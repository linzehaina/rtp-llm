package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;

import java.util.List;
import java.util.OptionalLong;

/**
 * Mode-specific part of delivery. Grouping chooses an ordered candidate group;
 * WorkerBatcher commits the prepared transaction, then invokes this strategy
 * outside the queue lock to submit a batch or publish individual routes.
 */
public interface DeliveryStrategy {

    /** Reserve the largest feasible prefix without mutating queue ownership. */
    DeliveryTransaction prepare(
            List<RequestRoute> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs);

    /** Fresh callback for GroupPlanner's strictly growing prefixes in one select call. */
    GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
            PrefillTimePredictor.Evaluator evaluator);

    /** Pure projection behavior paired with this live delivery strategy. */
    RouteProjection.DeliveryProjection projectionPolicy();

    /** Hand committed resources to the transport or publish individual routes outside the queue lock. */
    void deliver(DeliveryTransaction transaction, String decisionReason,
                 int remainingQueueDepth, WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator);

}
