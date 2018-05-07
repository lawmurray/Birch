/*
 * Delayed Beta-binomial random variate.
 */
class DelayBetaBinomial(x:Random<Integer>, n:Integer, ρ:DelayBeta) <
    DelayValue<Integer>(x) {
  /**
   * Number of trials.
   */
  n:Integer <- n;

  /**
   * Success probability.
   */
  ρ:DelayBeta <- ρ;

  function doSimulate() -> Integer {
    return simulate_beta_binomial(n, ρ.α, ρ.β);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function doCondition(x:Integer) {
    (ρ.α, ρ.β) <- update_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function pmf(x:Integer) -> Real {
    return pmf_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function cdf(x:Integer) -> Real {
    return cdf_beta_binomial(x, n, ρ.α, ρ.β);
  }
}

function DelayBetaBinomial(x:Random<Integer>, n:Integer, ρ:DelayBeta) ->
    DelayBetaBinomial {
  m:DelayBetaBinomial(x, n, ρ);
  return m;
}
