/*
 * Delayed Gaussian random variate.
 */
class DelayGaussian(x:Random<Real>&, μ:Real, σ2:Real) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Precision.
   */
  λ:Real <- 1.0/σ2;

  function simulate() -> Real {
    return simulate_gaussian(μ, 1.0/λ);
  }
  
  function observe(x:Real) -> Real {
    return observe_gaussian(x, μ, 1.0/λ);
  }
  
  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_gaussian(x, μ, 1.0/λ);
  }

  function cdf(x:Real) -> Real {
    return cdf_gaussian(x, μ, 1.0/λ);
  }
}

function DelayGaussian(x:Random<Real>&, μ:Real, σ2:Real) -> DelayGaussian {
  m:DelayGaussian(x, μ, σ2);
  return m;
}
