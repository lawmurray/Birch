/**
 * Normal-inverse-gamma distribution.
 *
 * This represents the joint distribution:
 *
 * $$\sigma^2 \sim \mathrm{Inverse-Gamma}(\alpha, \beta)$$
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, a^2\sigma^2),$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{Normal-Inverse-Gamma(\mu, a^2, \alpha, \beta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * In model code, it is not usual to use this final class directly. Instead,
 * establish the conjugate relationship via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, a^2*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` must appear in the
 * last argument of the distribution of `x`. The operation of `a2` on `σ2` may
 * be multiplication on the left (as above) or the right, or division on the
 * right.
 */
final class NormalInverseGamma(μ:Expression<Real>, a2:Expression<Real>,
    σ2:InverseGamma) < Distribution<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Precision scale.
   */
  λ:Expression<Real> <- 1.0/a2;
  
  /**
   * Variance.
   */
  σ2:InverseGamma <- σ2;
  
  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_normal_inverse_gamma(μ.value(), 1.0/λ.value(), σ2.α.value(), σ2.β.value());
  }

  function simulateLazy() -> Real? {
    return simulate_normal_inverse_gamma(μ.pilot(), 1.0/λ.pilot(), σ2.α.pilot(), σ2.β.pilot());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_normal_inverse_gamma(x, μ.value(), 1.0/λ.value(), σ2.α.value(), σ2.β.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_normal_inverse_gamma(x, μ, 1.0/λ, σ2.α, σ2.β);
  }

  function update(x:Real) {
    (σ2.α, σ2.β) <- box(update_normal_inverse_gamma(x, μ.value(), λ.value(), σ2.α.value(), σ2.β.value()));
  }

  function updateLazy(x:Expression<Real>) {
    (σ2.α, σ2.β) <- update_lazy_normal_inverse_gamma(x, μ, λ, σ2.α, σ2.β);
  }

  function downdate(x:Real) {
    (σ2.α, σ2.β) <- box(downdate_normal_inverse_gamma(x, μ.value(), λ.value(), σ2.α.value(), σ2.β.value()));
  }

  function cdf(x:Real) -> Real? {
    return cdf_normal_inverse_gamma(x, μ.value(), 1.0/λ.value(), σ2.α.value(), σ2.β.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_normal_inverse_gamma(P, μ.value(), 1.0/λ.value(), σ2.α.value(), σ2.β.value());
  }

  function graftNormalInverseGamma(compare:Distribution<Real>) ->
      NormalInverseGamma? {
    prune();
    if σ2 == compare {
      return this;
    } else {
      return nil;
    }
  }

  function link() {
    σ2.setChild(this);
  }
  
  function unlink() {
    σ2.releaseChild(this);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "NormalInverseGamma");
    buffer.set("μ", μ);
    buffer.set("a2", 1.0/λ);
    buffer.set("α", σ2.α);
    buffer.set("β", σ2.β);
  }
}

function NormalInverseGamma(μ:Expression<Real>, a2:Expression<Real>,
    σ2:InverseGamma) -> NormalInverseGamma {
  m:NormalInverseGamma(μ, a2, σ2);
  m.link();
  return m;
}
