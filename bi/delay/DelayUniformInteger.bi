/**
 * Delayed uniform integer random variate.
 */
class DelayUniformInteger(x:Random<Integer>, l:Integer, u:Integer) <
    DelayValue<Integer>(x) {
  /**
   * Lower bound.
   */
  l:Integer <- l;

  /**
   * Upper bound.
   */
  u:Integer <- u;

  function doSimulate() -> Integer {
    return simulate_uniform_int(l, u);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_uniform_int(x, l, u);
  }
}

function DelayUniformInteger(x:Random<Integer>, l:Integer, u:Integer) ->
    DelayUniformInteger {
  m:DelayUniformInteger(x, l, u);
  return m;
}
