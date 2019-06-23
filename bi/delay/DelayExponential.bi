/*
 * Delayed exponential random variate.
 */
final class DelayExponential(future:Real?, futureUpdate:Boolean, λ:Real) <
    DelayValue<Real>(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:Real <- λ;

  function simulate() -> Real {
    return simulate_exponential(λ);
  }

  function logpdf(x:Real) -> Real {
    return logpdf_exponential(x, λ);
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

function DelayExponential(future:Real?, futureUpdate:Boolean, λ:Real) ->
    DelayExponential {
  assert λ > 0.0;
  m:DelayExponential(future, futureUpdate, λ);
  return m;
}
