/**
 * Multivariate Gaussian distribution.
 */
class MultivariateGaussian<Type1,Type2>(μ:Type1, Σ:Type2) < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Type1 <- μ;
  
  /**
   * Covariance.
   */
  Σ:Type2 <- Σ;

  function size() -> Integer {
    return length(μ);
  }

  function update(μ:Type1, Σ:Type2) {
    this.μ <- μ;
    this.Σ <- Σ;
  }

  function doSimulate() -> Real[_] {
    return simulate_multivariate_gaussian(global.value(μ), global.value(Σ));
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_multivariate_gaussian(x, global.value(μ), global.value(Σ));
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) ->
    MultivariateGaussian<Real[_],Real[_,_]> {
  m:MultivariateGaussian<Real[_],Real[_,_]>(μ, Σ);
  m.initialize();
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_]) ->
    MultivariateGaussian<Expression<Real[_]>,Real[_,_]> {
  m:MultivariateGaussian<Expression<Real[_]>,Real[_,_]>(μ, Σ);
  m.initialize();
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian<Real[_],Expression<Real[_,_]>> {
  m:MultivariateGaussian<Real[_],Expression<Real[_,_]>>(μ, Σ);
  m.initialize();
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian<Expression<Real[_]>,Expression<Real[_,_]>> {
  m:MultivariateGaussian<Expression<Real[_]>,Expression<Real[_,_]>>(μ, Σ);
  m.initialize();
  return m;
}
