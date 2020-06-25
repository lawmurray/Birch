/**
 * Linear-normal-inverse-gamma-Gaussian distribution.
 */
final class LinearNormalInverseGammaGaussian(a:Expression<Real>,
    μ:NormalInverseGamma, c:Expression<Real>) < Distribution<Real> {
  /**
   * Scale.
   */
  a:Expression<Real> <- a;
    
  /**
   * Mean.
   */
  μ:NormalInverseGamma& <- μ;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return simulate_linear_normal_inverse_gamma_gaussian(a.value(),
        μ.μ.value(), 1.0/μ.λ.value(), c.value(), σ2.α.value(),
        σ2.β.value());
  }

  function simulateLazy() -> Real? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return simulate_linear_normal_inverse_gamma_gaussian(a.get(),
        μ.μ.get(), 1.0/μ.λ.get(), c.get(), σ2.α.get(), σ2.β.get());
  }
  
  function logpdf(x:Real) -> Real {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return logpdf_linear_normal_inverse_gamma_gaussian(x, a.value(),
        μ.μ.value(), 1.0/μ.λ.value(), c.value(), σ2.α.value(),
        σ2.β.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return logpdf_lazy_linear_normal_inverse_gamma_gaussian(x, a,
        μ.μ, 1.0/μ.λ, c, σ2.α, σ2.β);
  }

  function update(x:Real) {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    (μ.μ, μ.λ, σ2.α, σ2.β) <- box(update_linear_normal_inverse_gamma_gaussian(
        x, a.value(), μ.μ.value(), μ.λ.value(), c.value(), σ2.α.value(), 
        σ2.β.value()));
  }

  function updateLazy(x:Expression<Real>) {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    (μ.μ, μ.λ, σ2.α, σ2.β) <- update_lazy_linear_normal_inverse_gamma_gaussian(
        x, a, μ.μ, μ.λ, c, σ2.α, σ2.β);
  }

  function downdate(x:Real) {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    (μ.μ, μ.λ, σ2.α, σ2.β) <- box(downdate_linear_normal_inverse_gamma_gaussian(
        x, a.value(), μ.μ.value(), μ.λ.value(), c.value(), σ2.α.value(),
        σ2.β.value()));
  }

  function cdf(x:Real) -> Real? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return cdf_linear_normal_inverse_gamma_gaussian(x, a.value(),
        μ.μ.value(), 1.0/μ.λ.value(), c.value(), σ2.α.value(),
        σ2.β.value());
  }

  function quantile(P:Real) -> Real? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return quantile_linear_normal_inverse_gamma_gaussian(P, a.value(),
        μ.μ.value(), 1.0/μ.λ.value(), c.value(), σ2.α.value(),
        σ2.β.value());
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

function LinearNormalInverseGammaGaussian(a:Expression<Real>,
    μ:NormalInverseGamma, c:Expression<Real>) ->
    LinearNormalInverseGammaGaussian {
  m:LinearNormalInverseGammaGaussian(a, μ, c);
  m.link();
  return m;
}
