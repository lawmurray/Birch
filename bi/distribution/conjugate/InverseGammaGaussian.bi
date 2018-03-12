/**
 * Gaussian with conjugate prior on variance.
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
    σ2.update(σ2.k + 0.5, σ2.θ + 0.5*pow(value() - μ, 2.0));
  }

  function doRealize() {
    if (σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_gaussian(μ, σ2));
      } else {
        setWeight(observe_gaussian(value(), μ, σ2));
      }
    } else {
      ν:Real <- 2.0*σ2.k;         // degrees of freedom
      a:Real <- sqrt(σ2.θ/σ2.k);  // scale
      if (isMissing()) {
        set(simulate_student_t(ν)*a + μ);
      } else {
        setWeight(observe_student_t((value() - μ)/a, ν) - log(a));
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
