/*
 * Delayed discrete random variate.
 */
class DelayDiscrete(x:Random<Integer>&) < DelayValue<Integer>(x) {
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
