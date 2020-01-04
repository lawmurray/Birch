/*
 * ed linear-normal-inverse-gamma-Gaussian random variate where
 * components have independent and identical variance.
 */
final class LinearMultivariateNormalInverseGammaMultivariateGaussian(
    future:Real[_]?, futureUpdate:Boolean, A:Real[_,_],
    μ:MultivariateNormalInverseGamma, c:Real[_]) <
    Distribution<Real[_]>(future, futureUpdate) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Real[_] <- c;

  function simulate() -> Real[_] {
    return simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function update(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function downdate(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- downdate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }
}

function LinearMultivariateNormalInverseGammaMultivariateGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], μ:MultivariateNormalInverseGamma,
    c:Real[_]) -> LinearMultivariateNormalInverseGammaMultivariateGaussian {
  m:LinearMultivariateNormalInverseGammaMultivariateGaussian(future, futureUpdate, A, μ, c);
  μ.setChild(m);
  return m;
}
