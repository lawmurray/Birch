import distribution.Delay;
import math;
import random;

/**
 * Gaussian distribution.
 */
class Gaussian < Delay {
  /**
   * Value.
   */
  x:Real;
  
  /**
   * Weight.
   */
  w:Real;

  /**
   * Mean.
   */
  μ:Real;
  
  /**
   * Standard deviation.
   */
  σ:Real;

  /**
   * Value conversion.
   */
  function -> Real {
    value();
    return x;
  }
  
  function create(μ:Real, σ:Real) {
    assert(σ > 0.0);
    
    initialise();
    this.μ <- μ;
    this.σ <- σ;
  }

  function doSample() {
    cpp {{
    cast(this)->x = std::normal_distribution<double>(μ, σ)(rng);
    }}
  }

  function doObserve() {
    this.w <- -0.5*pow((x - μ)/σ, 2.0) - log(sqrt(2.0*π)*σ);
  }
  
  function set(x:Real) {
    this.x <- x;
  }
  
  function update(μ:Real, σ:Real) {
    assert(σ > 0.0);
    
    this.μ <- μ;
    this.σ <- σ;
  }
}

/**
 * Create.
 */
function Gaussian(μ:Real, σ:Real) -> Gaussian {
  m:Gaussian;
  m.create(μ, σ);
  return m;
}

/**
 * Simulate.
 */
function x:Real <~ m:Gaussian {
  m.sample();
  x <- m.x;
}

/**
 * Observe.
 */
function x:Real ~> m:Gaussian -> Real {
  m.set(x);
  m.observe();
  return m.w;
}

/**
 * Value assignment.
 */
function m:Gaussian <- x:Real {
  m.set(x);
}

/**
 * Initialise.
 */
function x:Gaussian ~ m:Gaussian {
  if (x.isRealised()) {
    x ~> m;
  }
  x <- m;
}
