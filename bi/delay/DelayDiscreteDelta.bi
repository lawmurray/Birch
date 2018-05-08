/*
 * Delayed delta function on a discrete random variate.
 */
class DelayDiscreteDelta(μ:DelayValue<Integer>) < DelayValue<Integer> {
  /**
   * Location.
   */
  μ:DelayValue<Integer> <- μ;

  function simulate() -> Integer {
    return simulate_delta(μ.simulate());
  }
  
  function observe(x:Integer) -> Real {
    return μ.observe(x);
  }

  function pmf(x:Integer) -> Real {
    return μ.pmf(x);
  }

  function cdf(x:Integer) -> Real {
    return μ.cdf(x);
  }
}

function DelayDiscreteDelta(μ:DelayValue<Integer>) -> DelayDiscreteDelta {
  m:DelayDiscreteDelta(μ);
  return m;
}
