/**
 * Normal-inverse-gamma-Gaussian random variable with delayed sampling.
 */
class DelayNormalInverseGammaGaussian(x:Random<Real>,
    μ:DelayNormalInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma <- μ;

  function doSimulate() -> Real {
    return simulate_normal_inverse_gamma_gaussian(μ.μ, μ.a2, μ.σ2.α, μ.σ2.β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_normal_inverse_gamma_gaussian(x, μ.μ, μ.a2, μ.σ2.α,
        μ.σ2.β);
  }

  function doCondition(x:Real) -> Real {
    (μ.μ, μ.a2, μ.σ2.α, μ.σ2.β) <- observe_normal_inverse_gamma_gaussian(x,
        μ.μ, μ.a2, μ.σ2.α, μ.σ2.β);
  }
}
