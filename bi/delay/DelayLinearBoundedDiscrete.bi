/*
 * Delayed delta function on a linear transformation of a bounded discrete
 * random variate.
 */
class DelayLinearBoundedDiscrete(x:Random<Integer>&, a:Integer,
    μ:DelayBoundedDiscrete, c:Integer) < DelayBoundedDiscrete(x, a*μ.l + c,
    a*μ.u + c) {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Integer <- a;
    
  /**
   * Location.
   */
  μ:DelayBoundedDiscrete <- μ;

  /**
   * Offset.
   */
  c:Integer <- c;

  function simulate() -> Integer {
    return simulate_delta(a*μ.simulate() + c);
  }
  
  function observe(x:Integer) -> Real {
    return μ.realize((x - c)/a);
  }

  function pmf(x:Integer) -> Real {
    return μ.pmf((x - c)/a);
  }

  function cdf(x:Integer) -> Real {
    return μ.cdf((x - c)/a);
  }
}

function DelayLinearBoundedDiscrete(x:Random<Integer>&, a:Integer,
    μ:DelayBoundedDiscrete, c:Integer) -> DelayLinearBoundedDiscrete {
  assert abs(a) == 1;
  m:DelayLinearBoundedDiscrete(x, a, μ, c);
  μ.setChild(m);
  return m;
}
