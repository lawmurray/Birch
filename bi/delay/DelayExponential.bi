/*
 * Delayed Exponential random variate.
 */
final class DelayExponential(x:Random<Real>&, λ:Real) < DelayValue<Real>(x) {
  /**
   * Rate.
   */
  λ:Real <- λ;

  function simulate() -> Real {
    return simulate_exponential(λ);
  }

  function observe(x:Real) -> Real {
    return observe_exponential(x, λ);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_exponential(x, λ);
  }

  function cdf(x:Real) -> Real {
    return cdf_exponential(x, λ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Exponential");
    buffer.set("λ", λ);
  }
}

function DelayExponential(x:Random<Real>&, λ:Real) -> DelayExponential {
  assert λ > 0;
  m:DelayExponential(x, λ);
  return m;
}
