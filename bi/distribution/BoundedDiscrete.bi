/*
 * Bounded discrete distribution.
 */
abstract class BoundedDiscrete < Discrete {
  abstract function lower() -> Integer?;
  abstract function upper() -> Integer?;

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    if !hasValue() {
      prune();
      graftFinalize();
      return this;
    } else {
      return nil;
    }
  }
}
