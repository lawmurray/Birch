/*
 * Delayed delta function on a discrete random variate.
 */
class DelayDiscreteDelta(x:Random<Integer>&, μ:DelayDiscrete) <
    DelayDiscrete(x) {
  /**
   * Location.
   */
  μ:DelayDiscrete <- μ;

  function simulate() -> Integer {
    return simulate_delta(μ.simulate());
  }
  
  function observe(x:Integer) -> Real {
    return μ.observe(x);
  }

  function pmf(x:Integer) -> Real {
    return μ.pmf(x);
  }

  function cdf(x:Integer) -> Real {
    return μ.cdf(x);
  }

  function lower() -> Integer? {
    return μ.lower();
  }
  
  function upper() -> Integer? {
    return μ.upper();
  }
}

function DelayDiscreteDelta(x:Random<Integer>&, μ:DelayDiscrete) ->
    DelayDiscreteDelta {
  m:DelayDiscreteDelta(x, μ);
  μ.setChild(m);
  return m;
}
