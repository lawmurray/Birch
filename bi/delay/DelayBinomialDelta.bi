/**
 * Binomial-delta random variable for delayed sampling.
 */
class DelayBinomialDelta(x:Random<Integer>, μ:DelayBinomial) <
    DelayValue<Integer>(x) {
  /**
   * Location.
   */
  μ:DelayBinomial <- μ;

  function doSimulate() -> Integer {
    return simulate_delta(μ.simulate());
  }
  
  function doObserve(x:Integer) -> Real {
    return μ.observe(x);
  }
}

function DelayBinomialDelta(x:Random<Integer>, μ:DelayBinomial) ->
    DelayBinomialDelta {
  m:DelayBinomialDelta(x, μ);
  return m;
}
