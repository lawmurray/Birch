/**
 * Beta-binomial random variable for delayed sampling.
 */
class DelayBetaBinomial(x:Random<Integer>, n:Integer, ρ:DelayBeta) < DelayValue<Integer>(x) {
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
}
