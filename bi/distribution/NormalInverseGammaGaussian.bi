/**
 * Normal-inverse-gamma-Gaussian distribution.
 */
final class NormalInverseGammaGaussian(μ:NormalInverseGamma) <
    Distribution<Real> {
  /**
   * Mean.
   */
  μ:NormalInverseGamma& <- μ;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return simulate_normal_inverse_gamma_gaussian(μ.μ.value(),
        1.0/μ.λ.value(), σ2.α.value(), σ2.β.value());
  }

  function simulateLazy() -> Real? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return simulate_normal_inverse_gamma_gaussian(μ.μ.get(),
        1.0/μ.λ.get(), σ2.α.get(), σ2.β.get());
  }
  
  function logpdf(x:Real) -> Real {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return logpdf_normal_inverse_gamma_gaussian(x, μ.μ.value(),
        1.0/μ.λ.value(), σ2.α.value(), σ2.β.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return logpdf_lazy_normal_inverse_gamma_gaussian(x, μ.μ,
        1.0/μ.λ, σ2.α, σ2.β);
  }

  function update(x:Real) {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    (μ.μ, μ.λ, σ2.α, σ2.β) <- box(update_normal_inverse_gamma_gaussian(
        x, μ.μ.value(), μ.λ.value(), σ2.α.value(), σ2.β.value()));
  }

  function updateLazy(x:Expression<Real>) {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    (μ.μ, μ.λ, σ2.α, σ2.β) <- update_lazy_normal_inverse_gamma_gaussian(
        x, μ.μ, μ.λ, σ2.α, σ2.β);
  }

  function downdate(x:Real) {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    (μ.μ, μ.λ, σ2.α, σ2.β) <- box(downdate_normal_inverse_gamma_gaussian(
        x, μ.μ.value(), μ.λ.value(), σ2.α.value(), σ2.β.value()));
  }

  function cdf(x:Real) -> Real? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return cdf_normal_inverse_gamma_gaussian(x, μ.μ.value(),
        1.0/μ.λ.value(), σ2.α.value(), σ2.β.value());
  }

  function quantile(P:Real) -> Real? {
    auto μ <- this.μ;
    auto σ2 <- μ.σ2;
    return quantile_normal_inverse_gamma_gaussian(P, μ.μ.value(),
        1.0/μ.λ.value(), σ2.α.value(), σ2.β.value());
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

function NormalInverseGammaGaussian(μ:NormalInverseGamma) ->
    NormalInverseGammaGaussian {
  m:NormalInverseGammaGaussian(μ);
  m.link();
  return m;
}
