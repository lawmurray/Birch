/**
 * Normal-inverse-gamma random variable with delayed sampling.
 *
 */
class DelayNormalInverseGamma(x:Random<Real>, μ:Real,
    σ2:DelayScaledInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;
  
  /**
   * Variance.
   */
  σ2:DelayScaledInverseGamma <- σ2;

  function doSimulate() -> Real {
    return simulate_normal_inverse_gamma(μ, σ2.a2, σ2.α, σ2.β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_normal_inverse_gamma(x, μ, σ2.a2, σ2.α, σ2.β);
  }
}
