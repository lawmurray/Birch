/*
 * Delayed Beta-binomial random variate.
 */
final class DelayBetaBinomial(future:Integer?, futureUpdate:Boolean,
    n:Integer, ρ:DelayBeta) < DelayBoundedDiscrete(future, futureUpdate, 0,
    n) {
  /**
   * Number of trials.
   */
  n:Integer <- n;

  /**
   * Success probability.
   */
  ρ:DelayBeta& <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_beta_binomial(n, ρ.α, ρ.β);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function update(x:Integer) {
    (ρ.α, ρ.β) <- update_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function downdate(x:Integer) {
    (ρ.α, ρ.β) <- downdate_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_beta_binomial(x, n, ρ.α, ρ.β);
  }
  
  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n;
  }
}

function DelayBetaBinomial(future:Integer?, futureUpdate:Boolean, n:Integer, ρ:DelayBeta) ->
    DelayBetaBinomial {
  m:DelayBetaBinomial(future, futureUpdate, n, ρ);
  ρ.setChild(m);
  return m;
}
