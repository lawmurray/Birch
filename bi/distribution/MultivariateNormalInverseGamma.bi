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
 *     x ~ Gaussian(μ, \Sigma*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` appears in the
 * last argument of the distribution of `x`. The operation of `\Sigma` on
 * `σ2` may be multiplication on left (as above) or right, or division
 * on right.
 */
class MultivariateNormalInverseGamma < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Real[_];

  /**
   * Variance.
   */
  Σ:Real[_,_];
  
  /**
   * Variance scale.
   */
  σ2:InverseGamma;

  function initialize(μ:Real[_], Σ:Real[_,_], σ2:InverseGamma) {
    super.initialize(σ2);
    this.μ <- μ;
    this.Σ <- Σ;
    this.σ2 <- σ2;
  }

  function update(μ:Real[_], Σ:Real[_,_]) {
    this.μ <- μ;
    this.Σ <- Σ;
  }

  function doCondition() {
    α:Real;
    β:Real;
    (α, β) <- update_multivariate_normal_inverse_gamma(value(), μ, Σ, σ2.α, σ2.β);
    σ2.update(α, β);
  }

  function doRealize() {
    if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_multivariate_gaussian(μ, Σ*σ2.value()));
      } else {
        setWeight(observe_multivariate_gaussian(value(), μ, Σ*σ2.value()));
      }
    } else {
      if (isMissing()) {
        set(simulate_multivariate_normal_inverse_gamma(μ, Σ, σ2.α, σ2.β));
      } else {
        setWeight(observe_multivariate_normal_inverse_gamma(value(), μ, Σ, σ2.α, σ2.β));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:MatrixScalarExpression) -> Random<Real[_]> {
  assert length(μ) == rows(Σ.A);
  assert length(μ) > 0;
  
  S:InverseGamma? <- InverseGamma?(Σ.x);
  if (S? && det(Σ.A) > 0.0) {
    x:MultivariateNormalInverseGamma;
    x.initialize(μ, Σ.A, S!);
    return x;
  } else {
    return Gaussian(μ, Σ.value());
  }
}
