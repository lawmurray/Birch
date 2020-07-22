/**
 * Discrete distribution.
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

  function graftDiscrete() -> Discrete? {
    prune();
    return this;
  }
}
