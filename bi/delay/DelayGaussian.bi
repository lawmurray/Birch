/**
 * Gaussian random variable with delayed sampling.
 */
class DelayGaussian(x:Random<Real>, μ:Real, σ2:Real) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:Real <- σ2;

  function doSimulate() -> Real {
    return simulate_gaussian(μ, σ2);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_gaussian(x, μ, σ2);
  }
}

function DelayGaussian(x:Random<Real>, μ:Real, σ2:Real) -> DelayGaussian {
  m:DelayGaussian(x, μ, σ2);
  return m;
}
