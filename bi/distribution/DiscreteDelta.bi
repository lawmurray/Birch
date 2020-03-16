/*
 * Delta distribution on a discrete distribution.
 */
final class DiscreteDelta(μ:Discrete) < Discrete {
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

  function graftFinalize() -> Boolean {
    if !μ.hasValue() {
      μ.setChild(this);
      return true;
    } else {
      return false;
    }
  }
}

function DiscreteDelta(μ:Discrete) -> DiscreteDelta {
  m:DiscreteDelta(μ);
  return m;
}
