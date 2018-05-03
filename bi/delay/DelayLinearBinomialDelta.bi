/**
 * Linear-binomial-Delta random variable for delayed sampling.
 */
class DelayLinearBinomialDelta(x:Random<Integer>, a:Integer, μ:DelayBinomial,
    c:Integer) < DelayValue<Integer>(x) {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Integer <- a;
    
  /**
   * Location.
   */
  μ:DelayBinomial <- μ;

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

function DelayLinearBinomialDelta(x:Random<Integer>, a:Integer,
    μ:DelayBinomial, c:Integer) -> DelayLinearBinomialDelta {
  assert abs(a) == 1;
  m:DelayLinearBinomialDelta(x, a, μ, c);
  return m;
}
