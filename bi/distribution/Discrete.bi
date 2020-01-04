/*
 * ed discrete random variate.
 */
abstract class Discrete(future:Integer?, futureUpdate:Boolean) <
    Distribution<Integer>(future, futureUpdate) {
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
