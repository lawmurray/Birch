/*
 * Delayed negative binomial random variate.
 */
class DelayNegativeBinomial(n:Integer, ρ:Real) < DelayValue<Integer> {
  /**
   * Number of successes before the experiment is stopped.
   */
  n:Integer <- n;

  /**
   * Success probability.
   */
  ρ:Real <- ρ;

  function simulate() -> Integer {
    return simulate_negative_binomial(n, ρ);
  }
  
  function observe(x:Integer) -> Real {
    return observe_negative_binomial(x, n, ρ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_negative_binomial(x, n, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_negative_binomial(x, n, ρ);
  }
}

function DelayNegativeBinomial(n:Integer, ρ:Real) -> DelayNegativeBinomial {
  m:DelayNegativeBinomial(n, ρ);
  return m;
}
