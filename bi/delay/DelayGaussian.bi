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
    return simulate_gaussian(μ.value(), 1.0/λ.value());
  }

  function simulatePilot() -> Real {
    return simulate_gaussian(μ.pilot(), 1.0/λ.pilot());
  }

  function simulatePropose() -> Real {
    return simulate_gaussian(μ.propose(), 1.0/λ.propose());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ.value(), 1.0/λ.value());
  }

  function logpdfPilot(x:Real) -> Real {
    return logpdf_gaussian(x, μ.pilot(), 1.0/λ.pilot());
  }

  function logpdfPropose(x:Real) -> Real {
    return logpdf_gaussian(x, μ.propose(), 1.0/λ.propose());
  }

  function lazy(x:Expression<Real>) -> Expression<Real>? {
    return lazy_gaussian(x, μ, 1.0/λ);
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
