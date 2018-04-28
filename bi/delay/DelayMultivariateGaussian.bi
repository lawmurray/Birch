/**
 * Multivariate Gaussian random variable with delayed sampling.
 */
class DelayMultivariateGaussian(x:Random<Real[_]>, μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>) < DelayValue<Real[_]>(x) {
  /**
   * Mean.
   */
  μ:Real[_] <- μ.value();

  /**
   * Covariance.
   */
  Σ:Real[_,_] <- Σ.value();

  function size() -> Integer {
    return length(μ);
  }

  function doSimulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ, Σ);
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_multivariate_gaussian(x, μ, Σ);
  }
  
  function doGraftMultivariateGaussian() -> DelayMultivariateGaussian? {
    prune();
    return this;
  }
}
