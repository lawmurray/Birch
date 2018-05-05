/**
 * Delayed gamma random variate.
 */
class DelayGamma(x:Random<Real>, k:Real, θ:Real) < DelayValue<Real>(x) {
  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  θ:Real <- θ;

  function doSimulate() -> Real {
    return simulate_gamma(k, θ);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_gamma(x, k, θ);
  }
}

function DelayGamma(x:Random<Real>, k:Real, θ:Real) -> DelayGamma {
  m:DelayGamma(x, k, θ);
  return m;
}
