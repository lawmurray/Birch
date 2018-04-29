/**
 * Uniform integer random variable with delayed sampling.
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
    return simulate_int_uniform(l, u);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_int_uniform(x, l, u);
  }
}

function DelayUniformInteger(x:Random<Integer>, l:Integer, u:Integer) ->
    DelayUniformInteger {
  m:DelayUniformInteger(x, l, u);
  return m;
}
