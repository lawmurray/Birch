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

  function construct(μ:Real, σ:Real) {
    assert(σ > 0.0);
    
    super.construct();
    this.μ <- μ;
    this.σ <- σ;
  }
  
  function construct() {
    super.construct();
  }

  function update(μ:Real, σ:Real) {
    assert(σ > 0.0);
    
    this.μ <- μ;
    this.σ <- σ;
  }

  /**
   * Value conversion.
   */
  function -> Real {
    if (isMissing()) {
      graft();
      realise();
    }
    return x;
  }
  
  function set(x:Real) {
    this.x <- x;
    this.missing <- false;
  }
  
  function sample() -> Real {
    cpp {{
    return std::normal_distribution<double>(μ, σ)(rng);
    }}
  }

  function observe(x:Real) -> Real {
    return -0.5*pow((x - μ)/σ, 2.0) - log(sqrt(2.0*π)*σ);
  }

  function doSample() {
    this.x <- sample();
  }
  
  function doObserve() {
    this.w <- observe(x);
  }
}

/**
 * Create.
 */
function Gaussian(μ:Real, σ:Real) -> Gaussian {
  m:Gaussian;
  m.construct(μ, σ);
  m.initialise();
  return m;
}

/**
 * Set.
 */
function m:Gaussian <- x:Real {
  m.set(x);
}

/**
 * Sample.
 */
function x:Real <~ m:Gaussian {
  m.graft();
  m.realise();
  x <- m.x;
}

/**
 * Observe.
 */
function x:Real ~> m:Gaussian -> Real {
  m.graft();
  m.set(x);
  m.realise();
  return m.w;
}

/**
 * Initialise.
 */
function x:Gaussian ~ m:Gaussian {
  assert(x.isUninitialised());
  
  if (!x.isMissing()) {
    m.set(x);
    m.graft();
    m.realise();
  }
  x <- m;
}
