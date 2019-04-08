/*
 * Delayed negative binomial random variate.
 */
class DelayNegativeBinomial(x:Random<Integer>&, n:Integer, ρ:Real) <
    DelayDiscrete(x) {
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
  
  function observe(x:Integer) -> Real {
    return observe_negative_binomial(x, n, ρ);
  }

  function update(x:Integer) {
    //
  }

  function downdate(x:Integer) {
    //
  }

  function pmf(x:Integer) -> Real {
    return pmf_negative_binomial(x, n, ρ);
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

function DelayNegativeBinomial(x:Random<Integer>&, n:Integer, ρ:Real) ->
    DelayNegativeBinomial {
  m:DelayNegativeBinomial(x, n, ρ);
  return m;
}
