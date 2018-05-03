/**
 * Linear-normal-inverse-gamma-log-Gaussian random variable with delayed
 * sampling.
 */
class DelayLinearNormalInverseGammaLogGaussian(x:Random<Real>, a:Real,
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
    return exp(simulate_linear_normal_inverse_gamma_gaussian(a, μ.μ, c, μ.a2,
        μ.σ2.α, μ.σ2.β));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_linear_normal_inverse_gamma_gaussian(log(x), a, μ.μ, c, 
        μ.a2, μ.σ2.α, μ.σ2.β) - log(x);
  }

  function doCondition(x:Real) {
    (μ.μ, μ.a2, μ.σ2.α, μ.σ2.β) <- update_linear_normal_inverse_gamma_gaussian(
        log(x), a, μ.μ, c, μ.a2, μ.σ2.α, μ.σ2.β);
  }
}

function DelayLinearNormalInverseGammaLogGaussian(x:Random<Real>, a:Real,
    μ:DelayNormalInverseGamma, c:Real) ->
    DelayLinearNormalInverseGammaLogGaussian {
  m:DelayLinearNormalInverseGammaLogGaussian(x, a, μ, c);
  return m;
}
