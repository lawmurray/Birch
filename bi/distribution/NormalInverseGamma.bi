/**
 * Normal inverse-gamma distribution. This represents the joint
 * distribution:
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
class NormalInverseGamma < Random<Real> {
  /**
   * Mean.
   */
  μ:Real;

  /**
   * Variance.
   */
  a2:Real;
  
  /**
   * Variance scale.
   */
  σ2:InverseGamma;

  function initialize(μ:Real, a2:Real, σ2:InverseGamma) {
    assert a2 > 0.0;
    
    super.initialize(σ2);
    this.μ <- μ;
    this.a2 <- a2;
    this.σ2 <- σ2;
  }

  function update(μ:Real, a2:Real) {
    assert a2 > 0.0;
    
    this.μ <- μ;
    this.a2 <- a2;
  }

  function doCondition() {
    α:Real;
    β:Real;
    (α, β) <- update_normal_inverse_gamma(value(), μ, a2, σ2.α, σ2.β);
    σ2.update(α, β);
  }

  function doSimulate() -> Real {
    if (σ2.isRealized()) {
      return simulate_gaussian(μ, a2*σ2.value());
    } else {
      return simulate_normal_inverse_gamma(μ, a2, σ2.α, σ2.β);
    }
  }

  function doObserve(x:Real) -> Real {
    if (σ2.isRealized()) {
      return observe_gaussian(x, μ, a2*σ2.value());
    } else {
      return observe_normal_inverse_gamma(x, μ, a2, σ2.α, σ2.β);
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:AffineExpression) -> Random<Real> {
  s2:InverseGamma? <- InverseGamma?(σ2.x);
  if (s2? && σ2.a > 0.0 && σ2.c == 0.0) {
    x:NormalInverseGamma;
    x.initialize(μ, σ2.a, s2!);
    return x;
  } else {
    return Gaussian(μ, σ2.value());
  }
}
