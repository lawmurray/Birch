/**
 * Affine-normal-inverse-gamma-log-Gaussian random variable with delayed
 * sampling.
 */
class DelayAffineNormalInverseGammaLogGaussian(x:Random<Real>, a:Real,
    μ:DelayNormalInverseGamma, c:Real) < DelayValue<Real>(x) {
  /**
   * Scale.
   */
  a:Real;
    
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function doSimulate() -> Real {
    return exp(simulate_affine_normal_inverse_gamma_gaussian(a, μ.μ, c, μ.a2,
        μ.σ2.α, μ.σ2.β));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_affine_normal_inverse_gamma_gaussian(log(x), a, μ.μ, c, 
        μ.a2, μ.σ2.α, μ.σ2.β) - log(x);
  }

  function doCondition(x:Real) -> Real {
    (μ.μ, μ.a2, μ.σ2.α, μ.σ2.β) <- update_affine_normal_inverse_gamma_gaussian(
        log(x), a, μ.μ, c, μ.a2, μ.σ2.α, μ.σ2.β);
  }
}
