/**
 * Log-Gaussian with Gaussian as its mean.
 */
class LogGaussianWithGaussianMean < LogGaussian {
  /**
   * Mean.
   */
  μ:Gaussian;

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

  function initialize(μ:Gaussian, σ2:Real) {
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
function LogGaussian(μ:Gaussian, σ2:Real) -> LogGaussian {
  x:LogGaussianWithGaussianMean;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Random<Real>, σ2:Real) -> LogGaussian {
  μ1:Gaussian? <- Gaussian?(μ);
  if (μ1?) {
    return LogGaussian(μ1!, σ2);
  } else {
    return LogGaussian(μ.value(), σ2);
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Gaussian, σ2:Real) -> LogGaussian {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Random<Real>, σ2:Real) -> LogGaussian {
  return LogGaussian(μ, σ2);
}
