/*
 * Delayed multivariate Gaussian random variate.
 */
class DelayMultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) <
    DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;

  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;

  function rows() -> Integer {
    return μ.rows();
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ, Σ);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_gaussian(x, μ, Σ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateGaussian");
    buffer.set("μ", μ.value());
    buffer.set("Σ", Σ.value());
  }
}

function DelayMultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) ->
    DelayMultivariateGaussian {
  m:DelayMultivariateGaussian(future, futureUpdate, μ, Σ);
  return m;
}
