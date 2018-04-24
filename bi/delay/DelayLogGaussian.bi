/**
 * Log-Gaussian random variable with delayed sampling.
 */
class DelayLogGaussian(x:Random<Real>, μ:Real, σ2:Real) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:Real <- σ2;

  function doSimulate() -> Real {
    return simulate_log_gaussian(μ, σ2);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_log_gaussian(x, μ, σ2);
  }
}
