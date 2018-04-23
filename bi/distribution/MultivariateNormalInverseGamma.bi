/**
 * Multivariate normal inverse-gamma distribution. This represents the joint
 * distribution:
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
class MultivariateNormalInverseGamma(μ:Expression<Real[_]>,
    Λ:Expression<Real[_,_]>, σ2:Expression<Real>) < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;

  /**
   * Precision.
   */
  Λ:Expression<Real[_,_]> <- Λ;
  
  /**
   * Variance scale.
   */
  σ2:Expression<Real> <- σ2;

  /*function doCondition() {
    α:Real;
    β:Real;
    (α, β) <- update_multivariate_normal_inverse_gamma(value(), μ, Λ, σ2.α, σ2.β);
    σ2.update(α, β);
  }*/

  function doSimulate() -> Real[_] {
    if (σ2.isRealized()) {
      return simulate_multivariate_gaussian(μ, inv(Λ)*σ2.value());
    } else {
      return simulate_multivariate_normal_inverse_gamma(μ, Λ, σ2.α, σ2.β);
    }
  }

  function doObserve(x:Real[_]) -> Real {
    if (σ2.isRealized()) {
      return observe_multivariate_gaussian(x, μ, inv(Λ)*σ2.value());
    } else {
      return observe_multivariate_normal_inverse_gamma(x, μ, Λ, σ2.α, σ2.β);
    }
  }
}

/**
 * Create Gaussian distribution.
 */
/*function Gaussian(μ:Real[_], Σ:MatrixScalarExpression) -> Random<Real[_]> {
  assert length(μ) == rows(Σ.A);
  assert length(μ) > 0;
  
  S:InverseGamma? <- InverseGamma?(Σ.x);
  if (S? && det(Σ.A) > 0.0) {
    x:MultivariateNormalInverseGamma;
    x.initialize(μ, inv(Σ.A), S!);
    return x;
  } else {
    return Gaussian(μ, Σ.value());
  }
}*/
