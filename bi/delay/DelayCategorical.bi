/**
 * Categorical random variable for delayed sampling.
 */
class DelayCategorical(x:Random<Integer>, ρ:Real[_]) <
    DelayValue<Integer>(x) {
  /**
   * Category probabilities.
   */
  ρ:Real[_] <- ρ;

  function doSimulate() -> Integer {
    return simulate_categorical(ρ);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_categorical(x, ρ);
  }
}

function DelayCategorical(x:Random<Integer>, ρ:Real[_]) -> DelayCategorical {
  m:DelayCategorical(x, ρ);
  return m;
}
