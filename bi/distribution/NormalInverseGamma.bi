/**
 * Normal inverse-gamma distribution. This represents the joint
 * distribution:
 *
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, a^2 \sigma^2),$$
 * $$\sigma^2 \sim \Gamma^{-1}(k, \theta).$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{N-}\Gamma^{-1}(\mu, a^2, k, \theta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * It is establihed via code such as the following:
 *
 *     σ2 ~ InverseGamma(k, θ);
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
   * Precision.
   */
  a2:Real;
  
  /**
   * Scale.
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
    σ2.update(σ2.α + 0.5, σ2.β + 0.5*pow(value() - μ, 2.0)/a2);
  }

  function doRealize() {
    if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_gaussian(μ, a2*σ2.value()));
      } else {
        setWeight(observe_gaussian(value(), μ, a2*σ2.value()));
      }
    } else {
      ν:Real <- 2.0*σ2.α;       // degrees of freedom
      s2:Real <- a2*σ2.β/σ2.α;  // squared scale
      if (isMissing()) {
        set(simulate_student_t(ν, μ, s2));
      } else {
        setWeight(observe_student_t(value(), ν, μ, s2));
      }
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
