/*
 * ed delta function on a linear transformation of a discrete random
 * variate.
 */
final class LinearDiscrete(a:Integer, μ:Discrete, c:Integer) < Discrete {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Integer <- a;
    
  /**
   * Location.
   */
  μ:Discrete& <- μ;

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
      return μ.logpdf((x - c)/a) - log(Real(abs(a)));
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

function LinearDiscrete(a:Integer, μ:Discrete, c:Integer) -> LinearDiscrete {
  assert abs(a) == 1;
  m:LinearDiscrete(a, μ, c);
  μ.setChild(m);
  return m;
}
