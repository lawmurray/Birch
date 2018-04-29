/**
 * Multivariate Gaussian random variable with delayed sampling.
 */
class DelayMultivariateGaussian(x:Random<Real[_]>, μ:Real[_], Σ:Real[_,_]) <
    DelayValue<Real[_]>(x) {
  /**
   * Mean.
   */
  μ:Real[_] <- μ;

  /**
   * Covariance.
   */
  Σ:Real[_,_] <- Σ;

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
    return this;
  }
}

function DelayMultivariateGaussian(x:Random<Real[_]>, μ:Real[_],
    Σ:Real[_,_]) -> DelayMultivariateGaussian {
  m:DelayMultivariateGaussian(x, μ, Σ);
  return m;
}
