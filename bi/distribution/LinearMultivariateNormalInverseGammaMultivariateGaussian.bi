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

  function rows() -> Integer {
    return c.rows();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_] {
    auto μ <- this.μ;
    return simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function simulateLazy() -> Real[_]? {
    auto μ <- this.μ;
    return simulate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        A.get(), μ.ν.get(), μ.Λ.get(), c.get(), μ.α.get(), μ.γ.get());
  }
  
  function logpdf(x:Real[_]) -> Real {
    auto μ <- this.μ;
    return logpdf_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value());
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    auto μ <- this.μ;
    return logpdf_lazy_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function update(x:Real[_]) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(update_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value()));
  }

  function updateLazy(x:Expression<Real[_]>) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_lazy_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function downdate(x:Real[_]) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(downdate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, A.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), μ.γ.value()));
  }

  function link() {
    auto μ <- this.μ;
    μ.setChild(this);
  }
  
  function unlink() {
    auto μ <- this.μ;
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
