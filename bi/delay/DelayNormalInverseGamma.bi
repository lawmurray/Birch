/**
 * Normal-inverse-gamma random variable with delayed sampling.
 *
 * This represents the joint distribution:
 *
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, a^2 \sigma^2),$$
 * $$\sigma^2 \sim \Gamma^{-1}(\alpha, \beta).$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{N-}\Gamma^{-1}(\mu, a^2, \alpha, \beta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * It is established via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, a2*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` appears in the
 * last argument of the distribution of `x`. The operation of `a2` on
 * `σ2` may be multiplication on left (as above) or right, or division
 * on right.
 */
class DelayNormalInverseGamma(x:Random<Real>, μ:Real, a2:Real,
    σ2:DelayInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  a2:Real <- a2;

  /**
   * Scale.
   */
  σ2:DelayInverseGamma <- σ2;

  function doSimulate() -> Real {
    return simulate_normal_inverse_gamma(μ, a2, σ2.α, σ2.β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_normal_inverse_gamma(x, μ, a2, σ2.α, σ2.β);
  }
}
