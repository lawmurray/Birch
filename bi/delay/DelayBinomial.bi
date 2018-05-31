/*
 * Delayed Binomial random variate.
 */
class DelayBinomial(x:Random<Integer>&, n:Integer, ρ:Real) <
    DelayValue<Integer>(x) {
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

  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n;
  }
}

function DelayBinomial(x:Random<Integer>&, n:Integer, ρ:Real) ->
    DelayBinomial {
  m:DelayBinomial(x, n, ρ);
  return m;
}
