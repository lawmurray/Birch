/**
 * Multivariate Gaussian distribution.
 */
class MultivariateGaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;

  function size() -> Integer {
    return length(μ);
  }

  function doSimulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ.value(), Σ.value());
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_multivariate_gaussian(x, μ.value(), Σ.value());
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) -> MultivariateGaussian {
  m:MultivariateGaussian(μ, Σ);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(μ, Literal(Σ));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>) -> MultivariateGaussian {
  return Gaussian(Literal(μ), Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(Literal(μ), Literal(Σ));
}
