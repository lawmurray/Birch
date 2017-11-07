/**
 * Gaussian with logaritm of a log-Gaussian as its mean.
 */
class GaussianWithLogMean < Gaussian {
  /**
   * Exponential of mean.
   */
  μ:LogGaussian;

  /**
   * Variance.
   */
  σ2:Real;
  
  /**
   * Prior (marginalized) mean.
   */
  μ_0:Real;
  
  /**
   * Prior (marginalized) variance.
   */
  σ2_0:Real;

  function initialize(μ:LogGaussian, σ2:Real) {
    super.initialize(μ);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doMarginalize() {
    μ_0 <- μ.μ;
    σ2_0 <- μ.σ2 + σ2;
    update(μ_0, σ2_0);
  }

  function doForward() {
    μ_0 <- μ.x;
    σ2_0 <- σ2;
    update(μ_0, σ2_0);
  }
  
  function doCondition() {
    k:Real <- μ.σ2/σ2_0;
    μ.update(μ.μ + k*(value() - μ_0), μ.σ2 - k*μ.σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:LogExpression, σ2:Real) -> Gaussian {
  x:LogGaussian? <- LogGaussian?(μ.x);
  if (x?) {
    y:GaussianWithLogMean;
    y.initialize(x!, σ2);
    return y;
  } else {
    return Gaussian(μ.value(), σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:LogExpression, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}
