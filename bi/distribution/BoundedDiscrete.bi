/*
 * Bounded discrete random variate.
 */
abstract class BoundedDiscrete < Discrete {
  abstract function lower() -> Integer?;
  abstract function upper() -> Integer?;
}
