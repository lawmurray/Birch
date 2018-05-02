/**
 * Offset-binomial-Delta random variable for delayed sampling.
 */
class DelayOffsetBinomialDelta(x:Random<Integer>, μ:DelayBinomial,
    c:Integer) < DelayValue<Integer>(x) {
  /**
   * Location.
   */
  μ:DelayBinomial <- μ;

  /**
   * Offset.
   */
  c:Integer <- c;

  function doSimulate() -> Integer {
    return simulate_delta(μ.simulate() + c);
  }
  
  function doObserve(x:Integer) -> Real {
    return μ.observe(x - c);
  }
}

function DelayOffsetBinomialDelta(x:Random<Integer>, μ:DelayBinomial,
    c:Integer) -> DelayOffsetBinomialDelta {
  m:DelayOffsetBinomialDelta(x, μ, c);
  return m;
}
