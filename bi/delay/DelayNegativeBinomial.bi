/*
 * Delayed negative binomial random variate.
 */
class DelayNegativeBinomial(x:Random<Integer>, n:Integer, ρ:Real) <
    DelayValue<Integer>(x) {
  /**
   * Number of successes before the experiment is stopped.
   */
  n:Integer <- n;

  /**
   * Success probability.
   */
  ρ:Real <- ρ;

  function doSimulate() -> Integer {
    return simulate_negative_binomial(n, ρ);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_negative_binomial(x, n, ρ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_negative_binomial(x, n, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_negative_binomial(x, n, ρ);
  }
}

function DelayNegativeBinomial(x:Random<Integer>, n:Integer, ρ:Real) ->
    DelayNegativeBinomial {
  m:DelayNegativeBinomial(x, n, ρ);
  return m;
}
