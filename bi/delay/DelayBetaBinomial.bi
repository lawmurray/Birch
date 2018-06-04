/*
 * Delayed Beta-binomial random variate.
 */
class DelayBetaBinomial(x:Random<Integer>&, n:Integer, ρ:DelayBeta) <
    DelayBoundedDiscrete(x, 0, n) {
  /**
   * Number of trials.
   */
  n:Integer <- n;

  /**
   * Success probability.
   */
  ρ:DelayBeta <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_beta_binomial(n, ρ.α, ρ.β);
    }
  }
  
  function observe(x:Integer) -> Real {
    return observe_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function condition(x:Integer) {
    (ρ.α, ρ.β) <- update_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function pmf(x:Integer) -> Real {
    return pmf_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function cdf(x:Integer) -> Real {
    return cdf_beta_binomial(x, n, ρ.α, ρ.β);
  }
  
  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n;
  }
}

function DelayBetaBinomial(x:Random<Integer>&, n:Integer, ρ:DelayBeta) ->
    DelayBetaBinomial {
  m:DelayBetaBinomial(x, n, ρ);
  ρ.setChild(m);
  return m;
}
