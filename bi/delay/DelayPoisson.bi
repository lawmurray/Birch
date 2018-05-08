/*
 * Delayed Poisson random variate.
 */
class DelayPoisson(λ:Real) < DelayValue<Integer> {
  /**
   * Rate.
   */
  λ:Real <- λ;

  function simulate() -> Integer {
    return simulate_poisson(λ);
  }
  
  function observe(x:Integer) -> Real {
    return observe_poisson(x, λ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_poisson(x, λ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_poisson(x, λ);
  }
}

function DelayPoisson(λ:Real) -> DelayPoisson {
  m:DelayPoisson(λ);
  return m;
}
