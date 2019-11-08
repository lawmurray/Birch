/**
 * Multivariate normal-inverse-gamma distribution.
 *
 * This represents the joint distribution:
 *
 * $$\sigma^2 \sim \mathrm{Inverse-Gamma}(\alpha, \beta)$$
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, Σ\sigma^2),$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{Normal-Inverse-Gamma(\mu, Σ, \alpha, \beta),$$
 *
 * and is a conjugate prior of a Gaussian distribution with both unknown mean
 * and variance. The variance scaling is independent and identical in the
 * sense that all components of $x$ share the same $\sigma^2$.
 *
 * In model code, it is not usual to use this final class directly. Instead,
 * establish the conjugate relationship via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, Σ*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` must appear in the
 * last argument of the distribution of `x`. The operation of `Σ` on `σ2` may
 * be multiplication on the left (as above) or the right, or division on the
 * right.
 */
final class MultivariateNormalInverseGamma(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>, α:Expression<Real>, β:Expression<Real>) <
    Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance scale.
   */
  Σ:Expression<Real[_,_]> <- Σ;

  /**
   * Covariance.
   */
  σ2:InverseGamma(α, β);
  
  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_multivariate_normal_inverse_gamma(μ, llt(Σ), σ2.α, σ2.β);
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_normal_inverse_gamma(x, μ, llt(Σ), σ2.α, σ2.β);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ,
          Σ, σ2.graftInverseGamma()!);
    }
  }
}
