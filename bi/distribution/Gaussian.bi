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

  function initialise(u:Gaussian) {
    super.initialise(u);
  }

  function initialise(μ:Real, σ:Real) {
    assert(σ > 0.0);
    
    super.initialise();
    this.μ <- μ;
    this.σ <- σ;
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
      if (!isRealised()) {
        realise();
      }
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
    return -0.5*pow((x - μ)/σ, 2.0) - log(σ) - 0.5*log(2.0*π);
  }

  function doSample() {
    set(sample());
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
  m.initialise(μ, σ);
  return m;
}

/**
 * Set.
 */
function m:Gaussian <- x:Real {
  m.set(x);
}

/**
 * Set from string.
 */
function m:Gaussian <- s:String {
  m.set(Real(s));
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
 * Sample.
 */
function x:Gaussian <~ m:Gaussian {
  assert(x.isUninitialised() && x.isMissing());
  m.graft();
  m.realise();
  x <- m;
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
 * Observe.
 */
function x:Gaussian ~> m:Gaussian {
  assert(x.isUninitialised() && !x.isMissing());
  m.graft();
  m.set(x.x);
  m.realise();
}

/**
 * Initialise.
 */
function x:Gaussian ~ m:Gaussian {
  assert(x.isUninitialised());
  if (!x.isMissing()) {
    x ~> m;
  }
  x <- m;
}
