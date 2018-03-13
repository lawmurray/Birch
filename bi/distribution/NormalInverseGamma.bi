/**
 * Normal inverse-gamma distribution. This represents the joint
 * distribution:
 *
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, α^2\sigma^2),$$
 * $$\sigma^2 \sim \Gamma^{-1}(k, \theta).$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{N-}\Gamma^{-1}(\mu, α^2, k, \theta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * It is establihed via code such as the following:
 *
 *     σ2 ~ InverseGamma(k, θ);
 *     x ~ Gaussian(μ, α2*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` appears in the
 * last argument of the distribution of `x`.
 */
class NormalInverseGamma < Random<Real> {
  /**
   * Mean.
   */
  μ:Real;

  /**
   * Variance.
   */
  α2:Real;
  
  /**
   * Scale.
   */
  σ2:InverseGamma;

  function initialize(μ:Real, α2:Real, σ2:InverseGamma) {
    super.initialize(σ2);
    this.μ <- μ;
    this.α2 <- α2;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    //σ2.update(σ2.k + 0.5, σ2.θ + 0.5*pow(value() - μ, 2.0));
  }

  function doRealize() {
    /*if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_gaussian(μ, σ2));
      } else {
        setWeight(observe_gaussian(value(), μ, σ2));
      }
    } else {
      if (isMissing()) {
        set(simulate_student_t(ν)*a + μ);
      } else {
        setWeight(observe_student_t((value() - μ)/a, ν) - log(a));
      }
    }*/
  }
}
