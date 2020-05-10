/**
 * Bounded discrete distribution.
 */
abstract class BoundedDiscrete < Discrete {
  abstract function lower() -> Integer?;
  abstract function upper() -> Integer?;

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    prune();
    return this;
  }
}
