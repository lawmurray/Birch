/*
 * Delayed binomial random variate.
 */
final class DelayBinomial(future:Integer?, futureUpdate:Boolean, n:Integer,
    ρ:Real) < DelayBoundedDiscrete(future, futureUpdate, 0, n) {
  /**
   * Number of trials.
   */
  n:Integer <- n;

  /**
   * Probability of success.
   */
  ρ:Real <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_binomial(n, ρ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_binomial(x, n, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_binomial(x, n, ρ);
  }

  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Binomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

function DelayBinomial(future:Integer?, futureUpdate:Boolean, n:Integer, ρ:Real) ->
    DelayBinomial {
  m:DelayBinomial(future, futureUpdate, n, ρ);
  return m;
}
