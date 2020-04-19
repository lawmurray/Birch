/*
 * Grafted linear-normal-inverse-gamma-Gaussian distribution where
 * components have independent and identical variance.
 */
final class LinearMultivariateNormalInverseGammaMultivariateGaussian(
    A:Expression<Real[_,_]>, μ:MultivariateNormalInverseGamma,
    c:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Scale.
   */
  A:Expression<Real[_,_]> <- A;

  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma <- μ;

  /**
   * Offset.
   */
  c:Expression<Real[_]> <- c;

  function simulate() -> Real[_] {
    return simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function update(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function downdate(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- downdate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function graftFinalize() -> Boolean {
    A.value();
    c.value();
    if !μ.hasValue() {
      link();
      return true;
    } else {
      return false;
    }
  }

  function link() {
    μ.setChild(this);
  }
  
  function unlink() {
    μ.releaseChild();
  }
}

function LinearMultivariateNormalInverseGammaMultivariateGaussian(
    A:Expression<Real[_,_]>, μ:MultivariateNormalInverseGamma,
    c:Expression<Real[_]>) ->
    LinearMultivariateNormalInverseGammaMultivariateGaussian {
  m:LinearMultivariateNormalInverseGammaMultivariateGaussian(A, μ, c);
  return m;
}
