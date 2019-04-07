/*
 * Delayed Beta random variate.
 */
class DelayBeta(x:Random<Real>&, α:Real, β:Real) < DelayValue<Real>(x) {
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

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_beta(x, α, β);
  }

  function cdf(x:Real) -> Real {
    return cdf_beta(x, α, β);
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function upper() -> Real? {
    return 1.0;
  }
}

function DelayBeta(x:Random<Real>&, α:Real, β:Real) -> DelayBeta {
  m:DelayBeta(x, α, β);
  return m;
}
