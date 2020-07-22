/**
 * Delta distribution on a discrete distribution.
 */
final class DiscreteDelta(μ:Discrete) < Discrete {
  /**
   * Location.
   */
  μ:Discrete <- μ;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer {
    if value? {
      return simulate_delta(value!);
    } else {
      return simulate_delta(μ.simulate());
    }
  }
  
//  function simulateLazy() -> Integer? {
//    if value? {
//      return simulate_delta(value!);
//    } else {
//      return simulate_delta(μ.simulateLazy()!);
//    }
//  }
  
  function logpdf(x:Integer) -> Real {
    if value? {
      return logpdf_delta(x, value!);
    } else {
      return μ.logpdf(x);
    }
  }

//  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
//    if value? {
//      return logpdf_lazy_delta(x, value!);
//    } else {
//      return μ.logpdfLazy(x);
//    }
//  }
  
  function update(x:Integer) {
    μ.clamp(x);
  }

//  function updateLazy(x:Expression<Integer>) {
//
//  }

  function cdf(x:Integer) -> Real? {
    return μ.cdf(x);
  }

  function lower() -> Integer? {
    return μ.lower();
  }
  
  function upper() -> Integer? {
    return μ.upper();
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

function DiscreteDelta(μ:Discrete) -> DiscreteDelta {
  m:DiscreteDelta(μ);
  m.link();
  return m;
}
