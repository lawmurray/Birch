/*
 * Log-Gaussian with cobjugate (logarithm of a) log-Gaussian prior on
 * mean.
 */
class LogGaussianLogGaussian < LogGaussian {
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
    μ_0 <- μ.value();
    σ2_0 <- σ2;
    update(μ_0, σ2_0);
  }
  
  function doCondition() {
    k:Real <- μ.σ2/σ2_0;
    μ.update(μ.μ + k*(log(value()) - μ_0), μ.σ2 - k*μ.σ2);
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:LogExpression, σ2:Real) -> LogGaussian {
  x:LogGaussian? <- LogGaussian?(μ.x);
  if (x?) {
    y:LogGaussianLogGaussian;
    y.initialize(x!, σ2);
    return y;
  } else {
    return LogGaussian(μ.value(), σ2);
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:LogExpression, σ2:Real) -> LogGaussian {
  return LogGaussian(μ, σ2);
}
