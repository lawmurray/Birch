/*
 * Delayed binomial random variate.
 */
class DelayBinomial(x:Random<Integer>&, n:Integer, ρ:Real) <
    DelayBoundedDiscrete(x, 0, n) {
  /**
   * Number of trials.
   */
  n:Integer <- n;

  /**
   * Probability of success.
   */
  ρ:Real <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_binomial(n, ρ);
    }
  }
  
  function observe(x:Integer) -> Real {
    return observe_binomial(x, n, ρ);
  }

  function update(x:Integer) {
    //
  }

  function downdate(x:Integer) {
    //
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
