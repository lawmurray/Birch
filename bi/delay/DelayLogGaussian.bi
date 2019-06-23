/*
 * Delayed log-Gaussian random variate.
 */
class DelayLogGaussian(future:Real?, futureUpdate:Boolean, μ:Real, σ2:Real) <
    DelayValue<Real>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  λ:Real <- 1.0/σ2;

  function simulate() -> Real {
    return simulate_log_gaussian(μ, 1.0/λ);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_log_gaussian(x, μ, 1.0/λ);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_log_gaussian(x, μ, 1.0/λ);
  }

  function cdf(x:Real) -> Real {
    return cdf_log_gaussian(x, μ, 1.0/λ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "LogGaussian");
    buffer.set("μ", μ);
    buffer.set("σ2", 1.0/λ);
  }
}

function DelayLogGaussian(future:Real?, futureUpdate:Boolean, μ:Real,
    σ2:Real) -> DelayLogGaussian {
  m:DelayLogGaussian(future, futureUpdate, μ, σ2);
  return m;
}
