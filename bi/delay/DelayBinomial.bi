/*
 * Delayed Binomial random variate.
 */
class DelayBinomial(n:Integer, ρ:Real) < DelayValue<Integer> {
  /**
   * Number of trials.
   */
  n:Integer <- n;

  /**
   * Probability of success.
   */
  ρ:Real <- ρ;

  function simulate() -> Integer {
    return simulate_binomial(n, ρ);
  }
  
  function observe(x:Integer) -> Real {
    return observe_binomial(x, n, ρ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_binomial(x, n, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_binomial(x, n, ρ);
  }
}

function DelayBinomial(n:Integer, ρ:Real) -> DelayBinomial {
  m:DelayBinomial(n, ρ);
  return m;
}
