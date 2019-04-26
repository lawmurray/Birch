/*
 * Bounded discrete random variate.
 */
class DelayBoundedDiscrete(future:Integer?, futureUpdate:Boolean, l:Integer,
    u:Integer) < DelayDiscrete(future, futureUpdate) {
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

/*
 * Constructor.
 */
function DelayBoundedDiscrete(future:Integer?, futureUpdate:Boolean,
    l:Integer, u:Integer) -> DelayBoundedDiscrete {
  m:DelayBoundedDiscrete(future, futureUpdate, l, u);
  return m;
}
