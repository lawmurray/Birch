/*
 * Delayed Beta-bernoulli random variate.
 */
class DelayBetaBernoulli(x:Random<Boolean>&, ρ:DelayBeta) <
    DelayValue<Boolean>(x) {
  /**
   * Success probability.
   */
  ρ:DelayBeta <- ρ;

  function simulate() -> Boolean {
    return simulate_beta_bernoulli(ρ.α, ρ.β);
  }
  
  function observe(x:Boolean) -> Real {
    return observe_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function condition(x:Boolean) {
    (ρ.α, ρ.β) <- update_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function pmf(x:Boolean) -> Real {
    return pmf_beta_bernoulli(x, ρ.α, ρ.β);
  }
}

function DelayBetaBernoulli(x:Random<Boolean>&, ρ:DelayBeta) ->
    DelayBetaBernoulli {
  m:DelayBetaBernoulli(x, ρ);
  ρ.setChild(m);
  return m;
}
