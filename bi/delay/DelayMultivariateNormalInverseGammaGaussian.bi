/*
 * Delayed multivariate normal-inverse-gamma-Gaussian random variate.
 */
final class DelayMultivariateNormalInverseGammaGaussian(x:Random<Real[_]>&,
    μ:DelayMultivariateNormalInverseGamma) < DelayValue<Real[_]>(x) {
  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma& <- μ;

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma_gaussian(μ!.μ, μ!.Λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_normal_inverse_gamma_gaussian(x, μ!.μ, μ!.Λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }

  function update(x:Real[_]) {
    (μ!.μ, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β) <- update_multivariate_normal_inverse_gamma_gaussian(
        x, μ!.μ, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function downdate(x:Real[_]) {
    (μ!.μ, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β) <- downdate_multivariate_normal_inverse_gamma_gaussian(
        x, μ!.μ, μ!.Λ, μ!.σ2!.α, μ!.σ2!.β);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_normal_inverse_gamma_gaussian(x, μ!.μ, μ!.Λ,
        μ!.σ2!.α, μ!.σ2!.β);
  }
}

function DelayMultivariateNormalInverseGammaGaussian(x:Random<Real[_]>&,
    μ:DelayMultivariateNormalInverseGamma) ->
    DelayMultivariateNormalInverseGammaGaussian {
  m:DelayMultivariateNormalInverseGammaGaussian(x, μ);
  μ.setChild(m);
  return m;
}
