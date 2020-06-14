/**
 * Delta function on a linear transformation of a bounded discrete
 * distribution.
 */
final class LinearBoundedDiscrete(a:Expression<Integer>, μ:BoundedDiscrete,
    c:Expression<Integer>) < BoundedDiscrete {
  /**
   * Scale. Should be 1 or -1 to ensure integer-invertible.
   */
  a:Expression<Integer> <- a;
    
  /**
   * Location.
   */
  μ:BoundedDiscrete <- μ;

  /**
   * Offset.
   */
  c:Expression<Integer> <- c;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer {
    if value? {
      return simulate_delta(value!);
    } else {
      return simulate_delta(a.value()*μ.simulate() + c.value());
    }
  }

//  function simulateLazy() -> Integer? {
//    if value? {
//      return simulate_delta(value!);
//    } else {
//      return simulate_delta(a.get()*μ.simulateLazy()! + c.get());
//    }
//  }

  function logpdf(x:Integer) -> Real {
    if value? {
      return logpdf_delta(x, value!);
    } else {
      return μ.logpdf((x - c.value())/a.value()) - log(abs(Real(a.value())));
    }
  }

//  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
//    if value? {
//      return logpdf_lazy_delta(x, value!);
//    } else {
//      return μ.logpdfLazy((x - c)/a) - log(abs(Real(a)));
//    }
//  }

  function update(x:Integer) {
    μ.clamp((x - c.value())/a.value());
  }

//  function updateLazy(x:Expression<Integer>) {
//
//  }

  function cdf(x:Integer) -> Real? {
    return μ.cdf((x - c.value())/a.value());
  }
  
  function lower() -> Integer? {
    auto a <- this.a.value();
    if a > 0 {
      return a*μ.lower()! + c.value();
    } else {
      return a*μ.upper()! + c.value();
    }
  }
  
  function upper() -> Integer? {
    auto a <- this.a.value();
    if a > 0 {
      return a*μ.upper()! + c.value();
    } else {
      return a*μ.lower()! + c.value();
    }
  }

  function link() {
    // clamp() used instead for discrete enumerations
    //μ.setChild(this);
  }
  
  function unlink() {
    // clamp() used instead for discrete enumerations
    //μ.releaseChild(this);
  }
}

function LinearBoundedDiscrete(a:Expression<Integer>, μ:BoundedDiscrete,
    c:Expression<Integer>) -> LinearBoundedDiscrete {
  m:LinearBoundedDiscrete(a, μ, c);
  m.link();
  return m;
}
