/*
 * Delayed delta function on a linear transformation of a discrete random
 * variate.
 */
final class DelayLinearDiscrete(x:Random<Integer>&, a:Integer,
    μ:DelayDiscrete, c:Integer) < DelayDiscrete(x) {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Integer <- a;
    
  /**
   * Location.
   */
  μ:DelayDiscrete& <- μ;

  /**
   * Offset.
   */
  c:Integer <- c;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_delta(a*μ!.simulate() + c);
    }
  }
  
  function observe(x:Integer) -> Real {
    assert !value?;
    return μ!.observe((x - c)/a);
  }
  
  function update(x:Integer) {
    μ!.clamp((x - c)/a);
  }

  function pmf(x:Integer) -> Real {
    return μ!.pmf((x - c)/a);
  }

  function cdf(x:Integer) -> Real {
    return μ!.cdf((x - c)/a);
  }

  function lower() -> Integer? {
    l:Integer? <- μ!.lower();
    if (l?) {
      l <- a*l! + c;
    }
    return l;
  }
  
  function upper() -> Integer? {
    u:Integer? <- μ!.upper();
    if (u?) {
      u <- a*u! + c;
    }
    return u;
  }
}

function DelayLinearDiscrete(x:Random<Integer>&, a:Integer,
    μ:DelayDiscrete, c:Integer) -> DelayLinearDiscrete {
  assert abs(a) == 1;
  m:DelayLinearDiscrete(x, a, μ, c);
  μ.setChild(m);
  return m;
}
