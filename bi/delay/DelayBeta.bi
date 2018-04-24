/**
 * Beta random variable with delayed sampling.
 */
class DelayBeta(x:Random<Real>, α:Real, β:Real) < DelayValue<Real>(x) {
  /**
   * First shape.
   */
  α:Real <- α;

  /**
   * Second shape.
   */
  β:Real <- β;

  function doSimulate() -> Real {
    return simulate_beta(α, β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_beta(x, α, β);
  }
}
