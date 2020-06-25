/**
 * Multivariate linear-normal-inverse-gamma-Gaussian distribution.
 */
final class LinearMultivariateNormalInverseGammaGaussian(
    a:Expression<Real[_]>, μ:MultivariateNormalInverseGamma,
    c:Expression<Real>) < Distribution<Real> {
  /**
   * Scale.
   */
  a:Expression<Real[_]> <- a;
    
  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;

  function rows() -> Integer {
    return c.rows();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    auto μ <- this.μ;
    return simulate_linear_multivariate_normal_inverse_gamma_gaussian(
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }

  function simulateLazy() -> Real? {
    auto μ <- this.μ;
    return simulate_linear_multivariate_normal_inverse_gamma_gaussian(
        a.get(), μ.ν.get(), μ.Λ.get(), c.get(), μ.α.get(), μ.γ.get());
        
  }
  
  function logpdf(x:Real) -> Real {
    auto μ <- this.μ;
    return logpdf_linear_multivariate_normal_inverse_gamma_gaussian(x,
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    auto μ <- this.μ;
    return logpdf_lazy_linear_multivariate_normal_inverse_gamma_gaussian(x,
        a, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function update(x:Real) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(update_linear_multivariate_normal_inverse_gamma_gaussian(
        x, a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), 
        μ.γ.value()));
  }

  function updateLazy(x:Expression<Real>) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- update_lazy_linear_multivariate_normal_inverse_gamma_gaussian(
        x, a, μ.ν, μ.Λ, c, μ.α, μ.γ);
  }

  function downdate(x:Real) {
    auto μ <- this.μ;
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(downdate_linear_multivariate_normal_inverse_gamma_gaussian(
        x, a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value()));
  }

  function cdf(x:Real) -> Real? {
    auto μ <- this.μ;
    return cdf_linear_multivariate_normal_inverse_gamma_gaussian(x,
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }

  function quantile(P:Real) -> Real? {
    auto μ <- this.μ;
    return quantile_linear_multivariate_normal_inverse_gamma_gaussian(P,
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
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

function LinearMultivariateNormalInverseGammaGaussian(a:Expression<Real[_]>,
    μ:MultivariateNormalInverseGamma, c:Expression<Real>) ->
    LinearMultivariateNormalInverseGammaGaussian {
  m:LinearMultivariateNormalInverseGammaGaussian(a, μ, c);
  m.link();
  return m;
}
