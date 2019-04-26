/*
 * Delayed delta function on a discrete random variate.
 */
final class DelayDiscreteDelta(future:Integer?, futureUpdate:Boolean,
    μ:DelayDiscrete) < DelayDiscrete(future, futureUpdate) {
  /**
   * Location.
   */
  μ:DelayDiscrete& <- μ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_delta(μ!.simulate());
    }
  }
  
  function observe(x:Integer) -> Real {
    assert !value?;
    return μ!.observe(x);
  }
  
  function update(x:Integer) {
    μ!.clamp(x);
  }

  function pmf(x:Integer) -> Real {
    return μ!.pmf(x);
  }

  function cdf(x:Integer) -> Real {
    return μ!.cdf(x);
  }

  function lower() -> Integer? {
    return μ!.lower();
  }
  
  function upper() -> Integer? {
    return μ!.upper();
  }
}

function DelayDiscreteDelta(future:Integer?, futureUpdate:Boolean,
    μ:DelayDiscrete) -> DelayDiscreteDelta {
  m:DelayDiscreteDelta(future, futureUpdate, μ);
  μ.setChild(m);
  return m;
}
