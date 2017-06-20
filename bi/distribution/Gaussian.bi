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

  /**
   * Value assignment.
   */
  operator <- x:Real {
    set(x);
  }

  /**
   * String assignment.
   */
  operator <- s:String {
    set(Real(s));
  }

  /**
   * Value conversion.
   */
  operator -> Real {
    if (isMissing()) {
      graft();
      realize();
    }
    return x;
  }

  function initialize(u:Gaussian) {
    super.initialize(u);
  }

  function initialize(μ:Real, σ2:Real) {
    assert(σ2 > 0.0);
    
    super.initialize();
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function update(μ:Real, σ2:Real) {
    assert(σ2 > 0.0);
    
    this.μ <- μ;
    this.σ2 <- σ2;
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
    w <- observe(x);
  }
}

/**
 * Create.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  m:Gaussian;
  m.initialize(μ, σ2);
  return m;
}

/**
 * Sample.
 */
operator x:Real <~ m:Gaussian {
  m.graft();
  m.realize();
  x <- m.x;
}

/**
 * Sample.
 */
operator x:Gaussian <~ m:Gaussian {
  assert(x.isUninitialized() && x.isMissing());
  m.graft();
  m.realize();
  x <- m;
}

/**
 * Observe.
 */
operator x:Real ~> m:Gaussian -> Real {
  m.graft();
  m.set(x);
  m.realize();
  return m.w;
}

/**
 * Observe.
 */
operator x:Gaussian ~> m:Gaussian {
  assert(x.isUninitialized() && !x.isMissing());
  m.graft();
  m.set(x.x);
  m.realize();
}

/**
 * Initialize.
 */
operator x:Gaussian ~ m:Gaussian {
  assert(x.isUninitialized());
  if (!x.isMissing()) {
    x ~> m;
  }
  x <- m;
}
