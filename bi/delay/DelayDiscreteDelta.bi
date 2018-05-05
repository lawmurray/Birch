/*
 * Delayed delta function on a discrete random variate.
 */
class DelayDiscreteDelta(x:Random<Integer>, μ:DelayValue<Integer>) <
    DelayValue<Integer>(x) {
  /**
   * Location.
   */
  μ:DelayValue<Integer> <- μ;

  function doSimulate() -> Integer {
    return simulate_delta(μ.simulate());
  }
  
  function doObserve(x:Integer) -> Real {
    return μ.observe(x);
  }
}

function DelayDiscreteDelta(x:Random<Integer>, μ:DelayValue<Integer>) ->
    DelayDiscreteDelta {
  m:DelayDiscreteDelta(x, μ);
  return m;
}
