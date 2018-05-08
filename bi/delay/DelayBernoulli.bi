/*
 * Delayed Bernoulli random variate.
 */
class DelayBernoulli(ρ:Real) < DelayValue<Boolean> {
  /**
   * Success probability.
   */
  ρ:Real <- ρ;

  function simulate() -> Boolean {
    return simulate_bernoulli(ρ);
  }
  
  function observe(x:Boolean) -> Real {
    return observe_bernoulli(x, ρ);
  }

  function pmf(x:Boolean) -> Real {
    return pmf_bernoulli(x, ρ);
  }
}

function DelayBernoulli(ρ:Real) -> DelayBernoulli {
  m:DelayBernoulli(ρ);
  return m;
}
