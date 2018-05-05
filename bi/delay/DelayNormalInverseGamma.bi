/*
 * Delayed normal-inverse-gamma random variate.
 */
class DelayNormalInverseGamma(x:Random<Real>, μ:Real, a2:Real,
    σ2:DelayInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;
  
  /**
   * Scale.
   */
  a2:Real <- a2;
  
  /**
   * Variance.
   */
  σ2:DelayInverseGamma <- σ2;

  function doSimulate() -> Real {
    return simulate_normal_inverse_gamma(μ, a2, σ2.α, σ2.β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_normal_inverse_gamma(x, μ, a2, σ2.α, σ2.β);
  }
}

function DelayNormalInverseGamma(x:Random<Real>, μ:Real, a2:Real,
    σ2:DelayInverseGamma) -> DelayNormalInverseGamma {
  m:DelayNormalInverseGamma(x, μ, a2, σ2);
  return m;
}
