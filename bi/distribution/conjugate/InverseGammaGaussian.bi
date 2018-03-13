/**
 * Gaussian with inverse-gamma prior on variance.
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
    σ2.update(σ2.α + 0.5, σ2.β + 0.5*pow(value() - μ, 2.0));
  }

  function doRealize() {
    if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_gaussian(μ, σ2.value()));
      } else {
        setWeight(observe_gaussian(value(), μ, σ2.value()));
      }
    } else {
      ν:Real <- 2.0*σ2.α;    // degrees of freedom
      s2:Real <- σ2.β/σ2.α;  // squared scale
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
