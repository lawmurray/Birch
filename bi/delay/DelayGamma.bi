/*
 * Delayed gamma random variate.
 */
class DelayGamma(k:Real, θ:Real) < DelayValue<Real> {
  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  θ:Real <- θ;

  function simulate() -> Real {
    return simulate_gamma(k, θ);
  }
  
  function observe(x:Real) -> Real {
    return observe_gamma(x, k, θ);
  }

  function pdf(x:Real) -> Real {
    return pdf_gamma(x, k, θ);
  }

  function cdf(x:Real) -> Real {
    return cdf_gamma(x, k, θ);
  }
}

function DelayGamma(k:Real, θ:Real) -> DelayGamma {
  m:DelayGamma(k, θ);
  return m;
}
