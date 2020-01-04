/*
 * ed delta function on a discrete random variate.
 */
final class DiscreteDelta(future:Integer?, futureUpdate:Boolean,
    μ:Discrete) < Discrete(future, futureUpdate) {
  /**
   * Location.
   */
  μ:Discrete& <- μ;

  function simulate() -> Integer {
    if value? {
      return simulate_delta(value!);
    } else {
      return simulate_delta(μ.simulate());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    if value? {
      return logpdf_delta(x, value!);
    } else {
      return μ.logpdf(x);
    }
  }
  
  function update(x:Integer) {
    μ.clamp(x);
  }

  function cdf(x:Integer) -> Real? {
    return μ.cdf(x);
  }

  function lower() -> Integer? {
    return μ.lower();
  }
  
  function upper() -> Integer? {
    return μ.upper();
  }
}

function DiscreteDelta(future:Integer?, futureUpdate:Boolean,
    μ:Discrete) -> DiscreteDelta {
  m:DiscreteDelta(future, futureUpdate, μ);
  μ.setChild(m);
  return m;
}
