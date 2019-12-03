/*
 * Delayed delta function on a linear transformation of a discrete random
 * variate.
 */
final class DelayLinearDiscrete(future:Integer?, futureUpdate:Boolean,
    a:Integer, μ:DelayDiscrete, c:Integer) < DelayDiscrete(future,
    futureUpdate) {
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

  function lower() -> Integer? {
    auto l <- μ.lower();
    if l? {
      l <- a*l! + c;
    }
    return l;
  }
  
  function upper() -> Integer? {
    auto u <- μ.upper();
    if u? {
      u <- a*u! + c;
    }
    return u;
  }
}

function DelayLinearDiscrete(future:Integer?, futureUpdate:Boolean, a:Integer,
    μ:DelayDiscrete, c:Integer) -> DelayLinearDiscrete {
  assert abs(a) == 1;
  m:DelayLinearDiscrete(future, futureUpdate, a, μ, c);
  μ.setChild(m);
  return m;
}
