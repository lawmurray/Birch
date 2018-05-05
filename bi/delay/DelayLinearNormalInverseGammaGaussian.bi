/*
 * Delayed linear-normal-inverse-gamma-Gaussian random variate.
 */
class DelayLinearNormalInverseGammaGaussian(x:Random<Real>, a:Real,
    μ:DelayNormalInverseGamma, c:Real) < DelayValue<Real>(x) {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma <- μ;

  /**
   * Offset.
   */
  c:Real <- c;

  function doSimulate() -> Real {
    return simulate_linear_normal_inverse_gamma_gaussian(a, μ.μ, c, μ.a2,
        μ.σ2.α, μ.σ2.β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_linear_normal_inverse_gamma_gaussian(x, a, μ.μ, c,
        μ.a2, μ.σ2.α, μ.σ2.β);
  }

  function doCondition(x:Real) {
    (μ.μ, μ.a2, μ.σ2.α, μ.σ2.β) <- update_linear_normal_inverse_gamma_gaussian(
        x, a, μ.μ, c, μ.a2, μ.σ2.α, μ.σ2.β);
  }
}

function DelayLinearNormalInverseGammaGaussian(x:Random<Real>, a:Real,
    μ:DelayNormalInverseGamma, c:Real) ->
    DelayLinearNormalInverseGammaGaussian {
  m:DelayLinearNormalInverseGammaGaussian(x, a, μ, c);
  return m;
}
