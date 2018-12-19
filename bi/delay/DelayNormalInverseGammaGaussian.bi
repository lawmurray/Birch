/*
 * Delayed normal-inverse-gamma-Gaussian random variate.
 */
class DelayNormalInverseGammaGaussian(x:Random<Real>&,
    μ:DelayNormalInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:DelayNormalInverseGamma& <- μ;

  function simulate() -> Real {
    return simulate_normal_inverse_gamma_gaussian(μ!.μ, μ!.a2, μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function observe(x:Real) -> Real {
    return observe_normal_inverse_gamma_gaussian(x, μ!.μ, μ!.a2, μ!.σ2!.α,
        μ!.σ2!.β);
  }

  function condition(x:Real) {
    (μ!.μ, μ!.a2, μ!.σ2!.α, μ!.σ2!.β) <- update_normal_inverse_gamma_gaussian(x,
        μ!.μ, μ!.a2, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Integer) -> Real {
    return pdf_normal_inverse_gamma_gaussian(x, μ!.μ, μ!.a2, μ!.σ2!.α, μ!.σ2!.β);
  }

  function cdf(x:Integer) -> Real {
    return cdf_normal_inverse_gamma_gaussian(x, μ!.μ, μ!.a2, μ!.σ2!.α, μ!.σ2!.β);
  }
}

function DelayNormalInverseGammaGaussian(x:Random<Real>&,
    μ:DelayNormalInverseGamma) -> DelayNormalInverseGammaGaussian {
  m:DelayNormalInverseGammaGaussian(x, μ);
  μ.setChild(m);
  return m;
}
