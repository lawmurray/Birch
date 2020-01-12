/*
 * ed multivariate normal-inverse-gamma-Gaussian random variate.
 */
final class MultivariateNormalInverseGammaMultivariateGaussian(
    μ:MultivariateNormalInverseGamma) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma& <- μ;

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        μ.ν, μ.Λ, μ.α, μ.γ);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_normal_inverse_gamma_multivariate_gaussian(x,
        μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function update(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function downdate(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- downdate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν, μ.Λ, μ.α, μ.γ);
  }
}

function MultivariateNormalInverseGammaMultivariateGaussian(
    μ:MultivariateNormalInverseGamma) ->
    MultivariateNormalInverseGammaMultivariateGaussian {
  m:MultivariateNormalInverseGammaMultivariateGaussian(μ);
  μ.setChild(m);
  return m;
}
