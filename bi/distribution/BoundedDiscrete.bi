/*
 * Bounded discrete random variate.
 */
abstract class BoundedDiscrete(future:Integer?, futureUpdate:Boolean,
    l:Integer, u:Integer) < Discrete(future, futureUpdate) {
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
