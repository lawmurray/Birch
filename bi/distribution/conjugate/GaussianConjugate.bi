/*
 * Gaussian with conjugate prior.
 */
class GaussianConjugate < Random<Real> {
  /**
   * Prior mean.
   */
  μ:Expression<Real>;

  /**
   * Prior variance.
   */
  σ2:Real;
  
  /**
   * Mean, after marginalizing out parents.
   */
  μ_m:Real;
  
  /**
   * Variance, after marginalizing out parents.
   */
  σ2_m:Real;
  
  /**
   * Mean, after conditioning on children.
   */
  μ_p:Real;
  
  /**
   * Variance, after conditioning on children.
   */
  σ2_p:Real;

  function initialize(μ:Expression<Real>, σ2:Real) {
    super.initialize(μ);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function isGaussian() -> Boolean {
    return true;
  }
  
  function updateGaussian(μ:Real, σ2:Real) {
    (μ_p, σ2_p) <- (μ, σ2);
  }
  
  function marginalizeGaussian(σ2:Real) -> (Real, Real) {
    return (μ_m, σ2_m + σ2);
  }
  
  function conditionGaussian(x:Real, μ_m:Real, σ2_m:Real) {
    (μ_p, σ2_p) <- update_gaussian_gaussian(x, μ_p, σ2_p, μ_m, σ2_m);
  }
  
  function doMarginalize() {
    if (μ.isGaussian() && !μ.isRealized()) {
      (μ_m, σ2_m) <- μ.marginalizeGaussian(σ2);
    } else {
      (μ_m, σ2_m) <- (μ.value(), σ2);
    }
    (μ_p, σ2_p) <- (μ_m, σ2_m);
  }
  
  function doCondition() {
    if (μ.isGaussian() && !μ.isRealized()) {
      μ.conditionGaussian(value(), μ_m, σ2_m);
    }
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_gaussian(μ_p, σ2_p));
    } else {
      setWeight(observe_gaussian(value(), μ_p, σ2_p));
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real) -> GaussianConjugate {
  x:GaussianConjugate;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Expression<Real>, σ2:Real) -> GaussianConjugate {
  return Gaussian(μ, σ2);
}
