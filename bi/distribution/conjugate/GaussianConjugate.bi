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
  
  function doMarginalize() {
    if (μ.isRealized()) {
      μ_m <- μ.value();
      σ2_m <- σ2;
    } else {
      μ1:Gaussian? <- Gaussian?(μ);
      μ2:GaussianConjugate? <- GaussianConjugate?(μ);
      if (μ1?) {
        μ_m <- μ1!.μ;
        σ2_m <- μ1!.σ2 + σ2;
      } else if (μ2?) {
        μ_m <- μ2!.μ_p;
        σ2_m <- μ2!.σ2_p + σ2;
      } else {
        μ_m <- μ.value();
        σ2_m <- σ2;
      }
    }
    μ_p <- μ_m;
    σ2_p <- σ2_m;
  }
  
  function doCondition() {
    μ1:Gaussian? <- Gaussian?(μ);
    μ2:GaussianConjugate? <- GaussianConjugate?(μ);
    if (μ1?) {
      (μ1!.μ, μ1!.σ2) <- update_gaussian_gaussian(value(), μ1!.μ, μ1!.σ2, μ_m, σ2_m);
    } else if (μ2?) {
      (μ2!.μ_p, μ2!.σ2_p) <- update_gaussian_gaussian(value(), μ2!.μ_p, μ2!.σ2_p, μ_m, σ2_m);
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
