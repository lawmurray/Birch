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
      return simulate_delta(value!);
    } else {
      return simulate_delta(a*μ.simulate() + c);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    if value? {
      return logpdf_delta(x, value!);
    } else {
      return μ.logpdf((x - c)/a) - log(abs(a));
    }
  }

  function update(x:Integer) {
    μ.clamp((x - c)/a);
  }

  function cdf(x:Integer) -> Real? {
    return μ.cdf((x - c)/a);
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
