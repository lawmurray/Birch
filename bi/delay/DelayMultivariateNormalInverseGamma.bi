/**
 * Multivariate normal-inverse-gamma random variable with delayed sampling.
 *
 * This represents the joint distribution:
 *
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, \Sigma, \sigma^2),$$
 * $$\sigma^2 \sim \Gamma^{-1}(\alpha, \beta).$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{N-}\Gamma^{-1}(\mu, \Sigma, \alpha, \beta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * It is established via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, Σ*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` appears in the
 * last argument of the distribution of `x`. The operation of `\Sigma` on
 * `σ2` may be multiplication on left (as above) or right, or division
 * on right.
 */
class DelayMultivariateNormalInverseGamma(x:Random<Real[_]>,
    μ:Expression<Real[_]>, Σ:TransformMultivariateScaledInverseGamma) <
    DelayValue<Real[_]>(x) {
  /**
   * Mean.
   */
  μ:Real[_] <- μ.value();

  /**
   * Precision.
   */
  Λ:Real[_,_] <- inv(Σ.A.value());

  /**
   * Scale.
   */
  σ2:DelayInverseGamma <- Σ.σ2;

  function doSimulate() -> Real[_] {
    return simulate_multivariate_normal_inverse_gamma(μ, Λ, σ2.α, σ2.β);
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_multivariate_normal_inverse_gamma(x, μ, Λ, σ2.α, σ2.β);
  }
}
