/*
 * Delayed Categorical random variate.
 */
class DelayCategorical(ρ:Real[_]) < DelayValue<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Real[_] <- ρ;

  function simulate() -> Integer {
    return simulate_categorical(ρ);
  }
  
  function observe(x:Integer) -> Real {
    return observe_categorical(x, ρ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_categorical(x, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_categorical(x, ρ);
  }
}

function DelayCategorical(ρ:Real[_]) -> DelayCategorical {
  m:DelayCategorical(ρ);
  return m;
}
