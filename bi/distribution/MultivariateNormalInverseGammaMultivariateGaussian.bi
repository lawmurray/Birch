/*
 * Grafted multivariate normal-inverse-gamma-Gaussian distribution.
 */
final class MultivariateNormalInverseGammaMultivariateGaussian(
    μ:MultivariateNormalInverseGamma) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma& <- μ;

  function rows() -> Integer {
    return μ.rows();
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value());
  }

  function update(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value());
  }

  function downdate(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- downdate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value());
  }
}

function MultivariateNormalInverseGammaMultivariateGaussian(
    μ:MultivariateNormalInverseGamma) ->
    MultivariateNormalInverseGammaMultivariateGaussian {
  m:MultivariateNormalInverseGammaMultivariateGaussian(μ);
  μ.setChild(m);
  return m;
}
