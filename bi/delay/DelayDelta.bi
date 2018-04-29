/**
 * Delta random variable for delayed sampling.
 */
class DelayDelta(x:Random<Integer>, μ:Integer) < DelayValue<Integer>(x) {
  /**
   * Location.
   */
  μ:Integer <- μ;

  function doSimulate() -> Integer {
    return simulate_delta(μ);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_delta(x, μ);
  }
}

function DelayDelta(x:Random<Integer>, μ:Integer) -> DelayDelta {
  m:DelayDelta(x, μ);
  return m;
}
