/*
 * Delayed Categorical random variate.
 */
class DelayCategorical(x:Random<Integer>&, ρ:Real[_]) <
    DelayValue<Integer>(x) {
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

function DelayCategorical(x:Random<Integer>&, ρ:Real[_]) -> DelayCategorical {
  m:DelayCategorical(x, ρ);
  return m;
}
