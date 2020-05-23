/**
 * Linear-normal-inverse-gamma-Gaussian distribution where
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
  μ:MultivariateNormalInverseGamma& <- μ;

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
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(update_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value()));
  }

  function downdate(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(downdate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value()));
  }

  function link() {
    μ.setChild(this);
  }
  
  function unlink() {
    μ.releaseChild(this);
  }
}

function LinearMultivariateNormalInverseGammaMultivariateGaussian(
    A:Expression<Real[_,_]>, μ:MultivariateNormalInverseGamma,
    c:Expression<Real[_]>) ->
    LinearMultivariateNormalInverseGammaMultivariateGaussian {
  m:LinearMultivariateNormalInverseGammaMultivariateGaussian(A, μ, c);
  m.link();
  return m;
}
