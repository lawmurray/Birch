/*
 * Bounded discrete random variate.
 */
abstract class BoundedDiscrete(l:Integer, u:Integer) < Discrete {
  /**
   * Lower bound
   */
  l:Integer <- l;

  /**
   * Upper bound.
   */
  u:Integer <- u;

  function lower() -> Integer? {
    return l;
  }
  
  function upper() -> Integer? {
    return u;
  }
}
