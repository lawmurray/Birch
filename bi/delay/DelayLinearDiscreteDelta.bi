/**
 * Delta function on a discrete random variate for delayed sampling.
 */
class DelayLinearDiscreteDelta(x:Random<Integer>, a:Integer,
    μ:DelayValue<Integer>, c:Integer) < DelayValue<Integer>(x) {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Integer <- a;
    
  /**
   * Location.
   */
  μ:DelayValue<Integer> <- μ;

  /**
   * Offset.
   */
  c:Integer <- c;

  function doSimulate() -> Integer {
    return simulate_delta(a*μ.simulate() + c);
  }
  
  function doObserve(x:Integer) -> Real {
    return μ.observe((x - c)/a);
  }
}

function DelayLinearDiscreteDelta(x:Random<Integer>, a:Integer,
    μ:DelayValue<Integer>, c:Integer) -> DelayLinearDiscreteDelta {
  assert abs(a) == 1;
  m:DelayLinearDiscreteDelta(x, a, μ, c);
  return m;
}
