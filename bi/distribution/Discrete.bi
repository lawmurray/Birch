/*
 * ed discrete random variate.
 */
abstract class Discrete < Distribution<Integer> {
  /**
   * Clamped value.
   */
  value:Integer?;

  /**
   * Clamp the value of the node.
   *
   * - x: The value.
   */
  function clamp(x:Integer) {
    assert !value? || value! == x;
    value <- x;
  }
}
