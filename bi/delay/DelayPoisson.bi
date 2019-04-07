/*
 * Delayed Poisson random variate.
 */
class DelayPoisson(x:Random<Integer>&, λ:Real) < DelayDiscrete(x) {
  /**
   * Rate.
   */
  λ:Real <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_poisson(λ);
    }
  }
  
  function observe(x:Integer) -> Real {
    return observe_poisson(x, λ);
  }

  function update(x:Integer) {
    //
  }

  function downdate(x:Integer) {
    //
  }

  function pmf(x:Integer) -> Real {
    return pmf_poisson(x, λ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_poisson(x, λ);
  }

  function lower() -> Integer? {
    return 0;
  }
}

function DelayPoisson(x:Random<Integer>&, λ:Real) -> DelayPoisson {
  m:DelayPoisson(x, λ);
  return m;
}
