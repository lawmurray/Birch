/*
 * Delayed inverse-gamma random variate.
 */
class DelayInverseGamma(x:Random<Real>&, α:Real, β:Real) <
    DelayValue<Real>(x) {
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

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_inverse_gamma(x, α, β);
  }

  function cdf(x:Real) -> Real {
    return cdf_inverse_gamma(x, α, β);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "InverseGamma");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

function DelayInverseGamma(x:Random<Real>&, α:Real, β:Real) ->
    DelayInverseGamma {
  m:DelayInverseGamma(x, α, β);
  return m;
}
