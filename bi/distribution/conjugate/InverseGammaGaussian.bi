/*
 * Gaussian with conjugate inverse-gamma prior on variance.
 */
class InverseGammaGaussian < Random<Real> {
  /**
   * Mean.
   */
  μ:Real;

  /**
   * Variance.
   */
  σ2:InverseGamma;

  function initialize(μ:Real, σ2:InverseGamma) {
    super.initialize(σ2);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    α:Real;
    β:Real;
    (α, β) <- update_inverse_gamma_gaussian(value(), μ, σ2.α, σ2.β);
    σ2.update(α, β);
  }

  function doRealize() {
    if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_gaussian(μ, σ2.value()));
      } else {
        setWeight(observe_gaussian(value(), μ, σ2.value()));
      }
    } else {
      if (isMissing()) {
        set(simulate_inverse_gamma_gaussian(μ, σ2.α, σ2.β));
      } else {
        setWeight(observe_inverse_gamma_gaussian(value(), μ, σ2.α, σ2.β));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:InverseGamma) -> InverseGammaGaussian {
  x:InverseGammaGaussian;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Random<Real>) -> Random<Real> {
  s2:InverseGamma? <- InverseGamma?(σ2);
  if (s2?) {
    return Gaussian(μ, s2!);
  } else {
    return Gaussian(μ, σ2.value());
  }
}
