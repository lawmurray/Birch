/**
 * Bernoulli random variable with delayed sampling.
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
