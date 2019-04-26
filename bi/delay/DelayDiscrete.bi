/*
 * Delayed discrete random variate.
 */
class DelayDiscrete(future:Integer?, futureUpdate:Boolean) <
    DelayValue<Integer>(future, futureUpdate) {
  /**
   * Clamped value.
   */
  value:Integer?;

  /**
   * Clamp the value of the node.
   *
   * - x: The value.
   */
  function clamp(x:Value) {
    assert !value? || value! == x;
    value <- x;
  }
}
