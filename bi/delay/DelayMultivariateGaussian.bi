/*
 * Delayed multivariate Gaussian random variate.
 */
class DelayMultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Real[_], Σ:Real[_,_]) < DelayValue<Real[_]>(future, futureUpdate) {
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

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ, Σ);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_gaussian(x, μ, Σ);
  }

  function update(x:Real[_]) {
    //
  }

  function downdate(x:Real[_]) {
    //
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_gaussian(x, μ, Σ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateGaussian");
    buffer.set("μ", μ);
    buffer.set("Σ", Σ);
  }
}

function DelayMultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Real[_], Σ:Real[_,_]) -> DelayMultivariateGaussian {
  m:DelayMultivariateGaussian(future, futureUpdate, μ, Σ);
  return m;
}
