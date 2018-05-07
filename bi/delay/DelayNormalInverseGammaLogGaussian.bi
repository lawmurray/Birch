/*
 * Delayed normal-inverse-gamma-log-Gaussian random variate.
 */
class DelayNormalInverseGammaLogGaussian(x:Random<Real>,
    μ:DelayNormalInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma <- μ;

  function doSimulate() -> Real {
    return exp(simulate_normal_inverse_gamma_gaussian(μ.μ, μ.a2, μ.σ2.α,
        μ.σ2.β));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_normal_inverse_gamma_gaussian(log(x), μ.μ, μ.a2, μ.σ2.α,
        μ.σ2.β) - log(x);
  }

  function doCondition(x:Real) {
    (μ.μ, μ.a2, μ.σ2.α, μ.σ2.β) <- update_normal_inverse_gamma_gaussian(
        log(x), μ.μ, μ.a2, μ.σ2.α, μ.σ2.β);
  }

  function pdf(x:Integer) -> Real {
    return pdf_normal_inverse_gamma_gaussian(log(x), μ.μ, μ.a2, μ.σ2.α,
        μ.σ2.β)/x;
  }

  function cdf(x:Integer) -> Real {
    return cdf_normal_inverse_gamma_gaussian(log(x), μ.μ, μ.a2, μ.σ2.α,
        μ.σ2.β);
  }
}

function DelayNormalInverseGammaLogGaussian(x:Random<Real>,
    μ:DelayNormalInverseGamma) -> DelayNormalInverseGammaLogGaussian {
  m:DelayNormalInverseGammaLogGaussian(x, μ);
  return m;
}
