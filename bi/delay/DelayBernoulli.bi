/**
 * Delayed Bernoulli random variate.
 */
class DelayBernoulli(x:Random<Boolean>, ρ:Real) < DelayValue<Boolean>(x) {
  /**
   * Success probability.
   */
  ρ:Real <- ρ;

  function doSimulate() -> Boolean {
    return simulate_bernoulli(ρ);
  }
  
  function doObserve(x:Boolean) -> Real {
    return observe_bernoulli(x, ρ);
  }
}

function DelayBernoulli(x:Random<Boolean>, ρ:Real) -> DelayBernoulli {
  m:DelayBernoulli(x, ρ);
  return m;
}
