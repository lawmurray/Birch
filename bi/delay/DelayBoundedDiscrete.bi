/*
 * Bounded discrete random variate.
 */
class DelayBoundedDiscrete(x:Random<Integer>&, l:Integer, u:Integer) <
    DelayDiscrete(x) {
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
function DelayBoundedDiscrete(x:Random<Integer>&, l:Integer, u:Integer) ->
    DelayBoundedDiscrete {
  m:DelayBoundedDiscrete(x, l, u);
  return m;
}
