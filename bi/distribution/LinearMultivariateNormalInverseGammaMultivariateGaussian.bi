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

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_] {
    return simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function simulateLazy() -> Real[_]? {
    return simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        A.get(), μ.ν.get(), μ.Λ.get(), c.get(), μ.α.get(), μ.γ.get());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    return logpdf_lazy_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function update(x:Real[_]) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(update_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value()));
  }

  function updateLazy(x:Expression<Real[_]>) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_lazy_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
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
