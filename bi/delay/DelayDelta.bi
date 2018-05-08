/*
 * Delayed delta random variate.
 */
class DelayDelta(μ:Integer) < DelayValue<Integer> {
  /**
   * Location.
   */
  μ:Integer <- μ;

  function simulate() -> Integer {
    return simulate_delta(μ);
  }
  
  function observe(x:Integer) -> Real {
    return observe_delta(x, μ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_delta(x, μ);
  }
}

function DelayDelta(μ:Integer) -> DelayDelta {
  m:DelayDelta(μ);
  return m;
}
