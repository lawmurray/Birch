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
   * Variance.
   */
  σ2:Real;

  function initialise(u:Gaussian) {
    super.initialise(u);
  }

  function initialise(μ:Real, σ2:Real) {
    assert(σ2 > 0.0);
    
    super.initialise();
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function update(μ:Real, σ2:Real) {
    assert(σ2 > 0.0);
    
    this.μ <- μ;
    this.σ2 <- σ2;
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
    return std::normal_distribution<double>(μ, ::sqrt(σ2))(rng);
    }}
  }

  function observe(x:Real) -> Real {
    return -0.5*(pow((x - μ), 2.0)/σ2 - log(σ2) - log(2.0*π));
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
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  m:Gaussian;
  m.initialise(μ, σ2);
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
