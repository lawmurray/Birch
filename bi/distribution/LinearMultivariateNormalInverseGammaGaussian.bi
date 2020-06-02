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

  function simulate() -> Real {
    return simulate_linear_multivariate_normal_inverse_gamma_gaussian(
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_linear_multivariate_normal_inverse_gamma_gaussian(x,
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }

  function update(x:Real) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(update_linear_multivariate_normal_inverse_gamma_gaussian(
        x, a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(), 
        μ.γ.value()));
  }

  function downdate(x:Real) {
    (μ.ν, μ.Λ, μ.α, μ.γ) <- box(downdate_linear_multivariate_normal_inverse_gamma_gaussian(
        x, a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value()));
  }

  function cdf(x:Real) -> Real? {
    return cdf_linear_multivariate_normal_inverse_gamma_gaussian(x,
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_linear_multivariate_normal_inverse_gamma_gaussian(P,
        a.value(), μ.ν.value(), μ.Λ.value(), c.value(), μ.α.value(),
        μ.γ.value());
  }

  function link() {
    μ.setChild(this);
  }
  
  function unlink() {
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
