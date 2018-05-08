/*
 * Delayed multivariate normal-inverse-gamma-Gaussian random variate.
 */
class DelayMultivariateInverseGammaGaussian(μ:Real[_],
    σ2:DelayInverseGamma) < DelayValue<Real[_]> {
  /**
   * Mean.
   */
  μ:Real[_] <- μ;

  /**
   * Variance.
   */
  σ2:DelayInverseGamma <- σ2;

  function simulate() -> Real[_] {
    return simulate_multivariate_inverse_gamma_gaussian(μ, σ2.α, σ2.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_inverse_gamma_gaussian(x, μ, σ2.α, σ2.β);
  }

  function condition(x:Real[_]) {
    (σ2.α, σ2.β) <- update_multivariate_inverse_gamma_gaussian(x, μ, σ2.α,
        σ2.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_inverse_gamma_gaussian(x, μ, σ2.α, σ2.β);
  }
}

function DelayMultivariateInverseGammaGaussian(μ:Real[_],
    σ2:DelayInverseGamma) -> DelayMultivariateInverseGammaGaussian {
  m:DelayMultivariateInverseGammaGaussian(μ, σ2);
  return m;
}
