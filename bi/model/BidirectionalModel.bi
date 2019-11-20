/**
 * Bidirectional model.
 */
abstract class BidirectionalModel < ForwardModel {
  /**
   * Simulate one step.
   *
   * - t: The step index. The caller guarantees that this constitutes either
   *      one greater than or one less than the `t` of the previous call to
   *      the same fiber, or 1 if the fiber has not been called on this
   *      before.
   */
  fiber simulate(t:Integer) -> Event {
    //
  }
}
