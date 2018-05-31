/*
 * Delayed log-Gaussian random variate.
 */
class DelayLogGaussian(x:Random<Real>&, μ:Real, σ2:Real) <
    DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:Real <- σ2;

  function simulate() -> Real {
    return simulate_log_gaussian(μ, σ2);
  }
  
  function observe(x:Real) -> Real {
    return observe_log_gaussian(x, μ, σ2);
  }

  function pdf(x:Real) -> Real {
    return pdf_log_gaussian(x, μ, σ2);
  }

  function cdf(x:Real) -> Real {
    return cdf_log_gaussian(x, μ, σ2);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayLogGaussian(x:Random<Real>&, μ:Real, σ2:Real) ->
    DelayLogGaussian {
  m:DelayLogGaussian(x, μ, σ2);
  return m;
}
