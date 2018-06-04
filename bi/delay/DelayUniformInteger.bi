/*
 * Delayed uniform integer random variate.
 */
class DelayUniformInteger(x:Random<Integer>&, l:Integer, u:Integer) <
    DelayBoundedDiscrete(x, l, u) {
  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_uniform_int(l, u);
    }
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

function DelayUniformInteger(x:Random<Integer>&, l:Integer, u:Integer) ->
    DelayUniformInteger {
  m:DelayUniformInteger(x, l, u);
  return m;
}
