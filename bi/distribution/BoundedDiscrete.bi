/*
 * Bounded discrete distribution.
 */
abstract class BoundedDiscrete < Discrete {
  abstract function lower() -> Integer?;
  abstract function upper() -> Integer?;
}
