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
final class NormalInverseGamma(future:Real?, futureUpdate:Boolean,
    μ:Real, a2:Real, σ2:InverseGamma) < Distribution<Real>(future,
    futureUpdate) {
  /**
   * Mean.
   */
  μ:Real <- μ;
  
  /**
   * Precision scale.
   */
  λ:Real <- 1.0/a2;
  
  /**
   * Variance.
   */
  σ2:InverseGamma& <- σ2;

  function simulate() -> Real {
    return simulate_normal_inverse_gamma(μ, 1.0/λ, σ2.α, σ2.β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_normal_inverse_gamma(x, μ, 1.0/λ, σ2.α, σ2.β);
  }

  function update(x:Real) {
    (σ2.α, σ2.β) <- update_normal_inverse_gamma(x, μ, λ, σ2.α, σ2.β);
  }

  function downdate(x:Real) {
    (σ2.α, σ2.β) <- downdate_normal_inverse_gamma(x, μ, λ, σ2.α, σ2.β);
  }

  function cdf(x:Real) -> Real? {
    return cdf_normal_inverse_gamma(x, μ, 1.0/λ, σ2.α, σ2.β);
  }

  function quantile(p:Real) -> Real? {
    return quantile_normal_inverse_gamma(p, μ, 1.0/λ, σ2.α, σ2.β);
  }

  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function graftNormalInverseGamma() -> NormalInverseGamma? {
    prune();
    return this;
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

function NormalInverseGamma(future:Real?, futureUpdate:Boolean, μ:Real,
    a2:Real, σ2:InverseGamma) -> NormalInverseGamma {
  m:NormalInverseGamma(future, futureUpdate, μ, a2, σ2);
  σ2.setChild(m);
  return m;
}
