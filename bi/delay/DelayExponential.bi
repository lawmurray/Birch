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

  function cdf(x:Real) -> Real {
    return cdf_exponential(x, λ);
  }

  function quantile(p:Real) -> Real? {
    return quantile_exponential(p, λ);
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
