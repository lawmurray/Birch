/*
 * Delayed negative binomial random variate.
 */
final class DelayNegativeBinomial(future:Integer?, futureUpdate:Boolean,
    n:Integer, ρ:Real) < DelayDiscrete(future, futureUpdate) {
  /**
   * Number of successes before the experiment is stopped.
   */
  n:Integer <- n;

  /**
   * Success probability.
   */
  ρ:Real <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_negative_binomial(n, ρ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_negative_binomial(x, n, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_negative_binomial(x, n, ρ);
  }

  function lower() -> Integer? {
    return 0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "NegativeBinomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

function DelayNegativeBinomial(future:Integer?, futureUpdate:Boolean,
    n:Integer, ρ:Real) -> DelayNegativeBinomial {
  m:DelayNegativeBinomial(future, futureUpdate, n, ρ);
  return m;
}
