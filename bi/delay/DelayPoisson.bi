/*
 * Delayed Poisson random variate.
 */
final class DelayPoisson(future:Integer?, futureUpdate:Boolean, λ:Real) <
    DelayDiscrete(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:Real <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_poisson(λ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_poisson(x, λ);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_poisson(x, λ);
  }

  function quantile(p:Real) -> Integer? {
    return quantile_poisson(p, λ);
  }

  function lower() -> Integer? {
    return 0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Poisson");
    buffer.set("λ", λ);
  }
}

function DelayPoisson(future:Integer?, futureUpdate:Boolean, λ:Real) ->
    DelayPoisson {
  m:DelayPoisson(future, futureUpdate, λ);
  return m;
}
