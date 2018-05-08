/*
 * Delayed Beta random variate.
 */
class DelayBeta(α:Real, β:Real) < DelayValue<Real> {
  /**
   * First shape.
   */
  α:Real <- α;

  /**
   * Second shape.
   */
  β:Real <- β;

  function simulate() -> Real {
    return simulate_beta(α, β);
  }
  
  function observe(x:Real) -> Real {
    return observe_beta(x, α, β);
  }

  function pdf(x:Real) -> Real {
    return pdf_beta(x, α, β);
  }

  function cdf(x:Real) -> Real {
    return cdf_beta(x, α, β);
  }
}

function DelayBeta(α:Real, β:Real) -> DelayBeta {
  m:DelayBeta(α, β);
  return m;
}
