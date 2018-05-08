/*
 * Delayed Gaussian random variate.
 */
class DelayGaussian(μ:Real, σ2:Real) < DelayValue<Real> {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:Real <- σ2;

  function simulate() -> Real {
    return simulate_gaussian(μ, σ2);
  }
  
  function observe(x:Real) -> Real {
    return observe_gaussian(x, μ, σ2);
  }

  function pdf(x:Real) -> Real {
    return pdf_gaussian(x, μ, σ2);
  }

  function cdf(x:Real) -> Real {
    return cdf_gaussian(x, μ, σ2);
  }
}

function DelayGaussian(μ:Real, σ2:Real) -> DelayGaussian {
  m:DelayGaussian(μ, σ2);
  return m;
}
