/*
 * Delayed inverse-gamma random variate.
 */
final class DelayInverseGamma(future:Real?, futureUpdate:Boolean, α:Real,
    β:Real) < DelayValue<Real>(future, futureUpdate) {
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
  
  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma(x, α, β);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
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

function DelayInverseGamma(future:Real?, futureUpdate:Boolean, α:Real,
    β:Real) -> DelayInverseGamma {
  m:DelayInverseGamma(future, futureUpdate, α, β);
  return m;
}
