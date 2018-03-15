/*
 * Multivariate Gaussian with conjugate inverse-gamma prior on diagonal
 * covariance.
 */
class MultivariateInverseGammaGaussian < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Real[_];

  /**
   * Variance.
   */
  σ2:InverseGamma;

  function initialize(μ:Real[_], σ2:InverseGamma) {
    super.initialize(σ2);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    α:Real;
    β:Real;
    (α, β) <- update_multivariate_inverse_gamma_gaussian(value(), μ, σ2.α, σ2.β);
    σ2.update(α, β);
  }

  function doRealize() {
    if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_multivariate_gaussian(μ, σ2.value()));
      } else {
        setWeight(observe_multivariate_gaussian(value(), μ, σ2.value()));
      }
    } else {
      if (isMissing()) {
        set(simulate_multivariate_inverse_gamma_gaussian(μ, σ2.α, σ2.β));
      } else {
        setWeight(observe_multivariate_inverse_gamma_gaussian(value(), μ, σ2.α, σ2.β));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:InverseGamma) -> MultivariateInverseGammaGaussian {
  x:MultivariateInverseGammaGaussian;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Random<Real>) -> Random<Real[_]> {
  s2:InverseGamma? <- InverseGamma?(σ2);
  if (s2?) {
    return Gaussian(μ, s2!);
  } else {
    return Gaussian(μ, identity(length(μ))*σ2.value());
  }
}
