/**
 * Multivariate normal-inverse-gamma-Gaussian distribution.
 */
final class MultivariateNormalInverseGammaMultivariateGaussian(
    μ:MultivariateNormalInverseGamma) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma& <- μ;

  function rows() -> Integer {
    auto μ <- this.μ;
    return μ.rows();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_] {
    auto μ <- this.μ;
    return simulate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value());
  }

  function simulateLazy() -> Real[_]? {
    auto μ <- this.μ;
    return simulate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        μ.ν.get(), μ.Λ.get(), μ.α.get(), μ.γ.get());
  }
  
  function logpdf(x:Real[_]) -> Real {
    auto μ <- this.μ;
    return logpdf_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value());
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    auto μ <- this.μ;
    return logpdf_lazy_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function update(x:Real[_]) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(update_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value()));
  }

  function updateLazy(x:Expression<Real[_]>) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_lazy_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν, μ.Λ, μ.α, μ.γ);
  }

  function downdate(x:Real[_]) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(downdate_multivariate_normal_inverse_gamma_multivariate_gaussian(
        x, μ.ν.value(), μ.Λ.value(), μ.α.value(), μ.γ.value()));
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

function MultivariateNormalInverseGammaMultivariateGaussian(
    μ:MultivariateNormalInverseGamma) ->
    MultivariateNormalInverseGammaMultivariateGaussian {
  m:MultivariateNormalInverseGammaMultivariateGaussian(μ);
  m.link();
  return m;
}
