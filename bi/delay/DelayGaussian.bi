/*
 * Delayed Gaussian random variate.
 */
class DelayGaussian(future:Real?, futureUpdate:Boolean, μ:Expression<Real>,
    σ2:Expression<Real>) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Mean.
   */
  auto μ <- μ;

  /**
   * Precision.
   */
  auto λ <- 1.0/σ2;

  function simulate() -> Real {
    return simulate_gaussian(μ, 1.0/λ);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ, 1.0/λ);
  }

  function logpdf(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_gaussian(x, μ, 1.0/λ);
  }
  
  function cdf(x:Real) -> Real? {
    return cdf_gaussian(x, μ, 1.0/λ);
  }

  function quantile(p:Real) -> Real? {
    return quantile_gaussian(p, μ, 1.0/λ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gaussian");
    buffer.set("μ", μ);
    buffer.set("σ2", 1.0/λ);
  }
}

function DelayGaussian(future:Real?, futureUpdate:Boolean,
    μ:Expression<Real>, σ2:Expression<Real>) -> DelayGaussian {
  m:DelayGaussian(future, futureUpdate, μ, σ2);
  return m;
}
