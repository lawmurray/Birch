/*
 * Delayed delta function on a linear transformation of a bounded discrete
 * random variate.
 */
final class DelayLinearBoundedDiscrete(future:Integer?, futureUpdate:Boolean,
    a:Integer, μ:DelayBoundedDiscrete, c:Integer) < DelayBoundedDiscrete(
    future, futureUpdate, a*μ.l + c, a*μ.u + c) {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Integer <- a;
    
  /**
   * Location.
   */
  μ:DelayBoundedDiscrete& <- μ;

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
}

function DelayLinearBoundedDiscrete(future:Integer?, futureUpdate:Boolean,
    a:Integer, μ:DelayBoundedDiscrete, c:Integer) ->
    DelayLinearBoundedDiscrete {
  assert abs(a) == 1;
  m:DelayLinearBoundedDiscrete(future, futureUpdate, a, μ, c);
  μ.setChild(m);
  return m;
}
