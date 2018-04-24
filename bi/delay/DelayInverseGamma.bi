/**
 * Inverse-gamma random variable with delayed sampling.
 */
class DelayInverseGamma(x:Random<Real>, α:Real, β:Real) < DelayValue<Real>(x) {
  /**
   * Shape.
   */
  α:Real <- α;

  /**
   * Scale.
   */
  β:Real <- β;

  function doSimulate() -> Real {
    return simulate_inverse_gamma(α, β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_inverse_gamma(x, α, β);
  }
}
