/**
 * Delayed Poisson random variate.
 */
class DelayPoisson(x:Random<Integer>, λ:Real) < DelayValue<Integer>(x) {
  /**
   * Rate.
   */
  λ:Real <- λ;

  function doSimulate() -> Integer {
    return simulate_poisson(λ);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_poisson(x, λ);
  }
}

function DelayPoisson(x:Random<Integer>, λ:Real) -> DelayPoisson {
  m:DelayPoisson(x, λ);
  return m;
}
