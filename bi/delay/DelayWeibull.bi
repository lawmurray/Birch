/*
 * Delayed Weibull random variate.
 */
final class DelayWeibull(future:Real?, futureUpdate:Boolean, k:Real, λ:Real) <
    DelayValue<Real>(future, futureUpdate) {
  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  λ:Real <- λ;

  function simulate() -> Real {
    return simulate_weibull(k, λ);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_weibull(x, k, λ);
  }

  function cdf(x:Real) -> Real {
    return cdf_weibull(x, k, λ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Weibull");
    buffer.set("k", k);
    buffer.set("λ", λ);
  }
}

function DelayWeibull(future:Real?, futureUpdate:Boolean, k:Real, λ:Real) ->
    DelayWeibull {
  m:DelayWeibull(future, futureUpdate, k, λ);
  return m;
}
