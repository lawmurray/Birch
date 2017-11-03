/**
 * Log-Gaussian distribution.
 */
class LogGaussian < Random<Real> {
  /**
   * Mean after log transformation.
   */
  μ:Real;
  
  /**
   * Variance after log transformation.
   */
  σ2:Real;

  function initialize(u:Gaussian) {
    super.initialize(u);
  }

  function initialize(u:LogGaussian) {
    super.initialize(u);
  }

  function initialize(μ:Real, σ2:Real) {
    super.initialize();
    update(μ, σ2);
  }

  function update(μ:Real, σ2:Real) {
    assert σ2 >= 0.0;
    
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_log_gaussian(μ, σ2));
    } else {
      setWeight(observe_log_gaussian(x, μ, σ2));
    }
  }
}

/**
 * Synonym for LogGaussian.
 */
type LogNormal = LogGaussian;

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Real) -> LogGaussian {
  m:LogGaussian;
  m.initialize(μ, σ2);
  return m;
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Real, σ2:Real) -> LogGaussian {
  return LogGaussian(μ, σ2);
}
