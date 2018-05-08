/*
 * Delayed multivariate normal-inverse-gamma-Gaussian random variate.
 */
class DelayMultivariateNormalInverseGammaGaussian(
    μ:DelayMultivariateNormalInverseGamma) < DelayValue<Real[_]> {
  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma <- μ;

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma_gaussian(μ.μ, μ.Λ,
        μ.σ2.α, μ.σ2.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_normal_inverse_gamma_gaussian(x, μ.μ, μ.Λ,
        μ.σ2.α, μ.σ2.β);
  }

  function condition(x:Real[_]) {
    (μ.μ, μ.Λ, μ.σ2.α, μ.σ2.β) <- update_multivariate_normal_inverse_gamma_gaussian(
        x, μ.μ, μ.Λ, μ.σ2.α, μ.σ2.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_normal_inverse_gamma_gaussian(x, μ.μ, μ.Λ,
        μ.σ2.α, μ.σ2.β);
  }
}

function DelayMultivariateNormalInverseGammaGaussian(
    μ:DelayMultivariateNormalInverseGamma) ->
    DelayMultivariateNormalInverseGammaGaussian {
  m:DelayMultivariateNormalInverseGammaGaussian(μ);
  return m;
}
