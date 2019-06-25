/**
 * Multivariate normal-inverse-gamma distribution.
 *
 * This represents the joint distribution:
 *
 * $$\sigma^2 \sim \mathrm{Inverse-Gamma}(\alpha, \beta)$$
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, A\sigma^2),$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{Normal-Inverse-Gamma(\mu, A, \alpha, \beta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * In model code, it is not usual to use this final class directly. Instead,
 * establish the conjugate relationship via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, A*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` must appear in the
 * last argument of the distribution of `x`. The operation of `A` on `σ2` may
 * be multiplication on the left (as above) or the right, or division on the
 * right.
 */
final class MultivariateNormalInverseGamma(μ:Expression<Real[_]>,
    A:Expression<Real[_,_]>, α:Expression<Real>, β:Expression<Real>) <
    Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance scale.
   */
  A:Expression<Real[_,_]> <- A;

  /**
   * Variance.
   */
  σ2:InverseGamma(α, β);
  
  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_multivariate_gaussian(μ, A*σ2.value());
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_gaussian(x, μ, A*σ2.value());
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ,
          A, σ2.graftInverseGamma()!);
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "MultivariateNormalInverseGamma");
      buffer.set("μ", μ);
      buffer.set("A", A);
      buffer.set("α", σ2.α);
      buffer.set("β", σ2.β);
    }
  }
}
