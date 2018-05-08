/*
 * Delayed inverse-gamma random variate.
 */
class DelayInverseGamma(α:Real, β:Real) < DelayValue<Real> {
  /**
   * Shape.
   */
  α:Real <- α;

  /**
   * Scale.
   */
  β:Real <- β;

  function simulate() -> Real {
    return simulate_inverse_gamma(α, β);
  }
  
  function observe(x:Real) -> Real {
    return observe_inverse_gamma(x, α, β);
  }

  function pdf(x:Real) -> Real {
    return pdf_inverse_gamma(x, α, β);
  }

  function cdf(x:Real) -> Real {
    return cdf_inverse_gamma(x, α, β);
  }
}

function DelayInverseGamma(α:Real, β:Real) -> DelayInverseGamma {
  m:DelayInverseGamma(α, β);
  return m;
}
