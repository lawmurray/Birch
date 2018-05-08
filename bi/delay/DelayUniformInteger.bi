/*
 * Delayed uniform integer random variate.
 */
class DelayUniformInteger(l:Integer, u:Integer) < DelayValue<Integer> {
  /**
   * Lower bound.
   */
  l:Integer <- l;

  /**
   * Upper bound.
   */
  u:Integer <- u;

  function simulate() -> Integer {
    return simulate_uniform_int(l, u);
  }
  
  function observe(x:Integer) -> Real {
    return observe_uniform_int(x, l, u);
  }

  function pmf(x:Integer) -> Real {
    return pmf_uniform_int(x, l, u);
  }

  function cdf(x:Integer) -> Real {
    return cdf_uniform_int(x, l, u);
  }
}

function DelayUniformInteger(l:Integer, u:Integer) ->
    DelayUniformInteger {
  m:DelayUniformInteger(l, u);
  return m;
}
