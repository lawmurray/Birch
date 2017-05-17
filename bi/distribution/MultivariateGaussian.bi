import distribution.Delay;
import distribution.Gaussian;
import math;
import random;

/**
 * Multivariate Gaussian distribution.
 */
class MultivariateGaussian(D1:Integer) < Delay {
  /**
   * Number of dimensions.
   */
  D:Integer <- D1;

  /**
   * Value.
   */
  x:Real[D];
  
  /**
   * Weight.
   */
  w:Real;

  /**
   * Mean.
   */
  μ:Real[D];
  
  /**
   * Covariance matrix.
   */
  Σ:Real[D,D];

  function initialize(u:MultivariateGaussian) {
    super.initialize(u);
  }

  function initialize(μ:Real[_], Σ:Real[_,_]) {
    super.initialize();
    update(μ, Σ);
  }

  function update(μ:Real[_], Σ:Real[_,_]) {
    this.μ <- μ;
    this.Σ <- Σ;
  }

  /**
   * Value conversion.
   */
  function -> Real[_] {
    if (isMissing()) {
      graft();
      realize();
    }
    return x;
  }
  
  function set(x:Real[_]) {
    assert(length(x) == D);
    
    this.x <- x;
    this.missing <- false;
  }
  
  function sample() -> Real[_] {
    x:Real[D];
    d:Integer;
    for (d in 1..D) {
      x[d] <~ Gaussian(0.0, 1.0);
    }
    return μ + llt(Σ)*x;
  }

  function observe(x:Real[_]) -> Real {
    L:Real[D,D];
    L <- llt(Σ);
    return -0.5*squaredNorm(solve(L, x - μ)) - log(determinant(L)) - 0.5*Real(D)*log(2.0*π);
  }

  function doSample() {
    set(sample());
  }
  
  function doObserve() {
    this.w <- observe(x);
  }
  
  function copy(o:MultivariateGaussian) {
    super.copy(o);
    this.x <- o.x;
    this.w <- o.w;
    this.μ <- o.μ;
    this.Σ <- o.Σ;
  }
}

/**
 * Create.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  D:Integer <- length(μ);
  assert(rows(Σ) == D);
  assert(columns(Σ) == D);
  m:MultivariateGaussian(D);
  m.initialize(μ, Σ);
  return m;
}

/**
 * Set.
 */
function m:MultivariateGaussian <- x:Real[_] {
  m.set(x);
}

/**
 * Sample.
 */
function x:Real[_] <~ m:MultivariateGaussian {
  assert(length(x) == m.D);
  m.graft();
  m.realize();
  x <- m.x;
}

/**
 * Sample.
 */
function x:MultivariateGaussian <~ m:MultivariateGaussian {
  assert(x.isUninitialized() && x.isMissing());
  m.graft();
  m.realize();
  x <- m;
}

/**
 * Observe.
 */
function x:Real[_] ~> m:MultivariateGaussian -> Real {
  assert(length(x) == m.D);
  m.graft();
  m.set(x);
  m.realize();
  return m.w;
}

/**
 * Observe.
 */
function x:MultivariateGaussian ~> m:MultivariateGaussian {
  assert(x.isUninitialized() && !x.isMissing());
  m.graft();
  m.set(x.x);
  m.realize();
}

/**
 * Initialise.
 */
function x:MultivariateGaussian ~ m:MultivariateGaussian {
  assert(x.isUninitialized());
  if (!x.isMissing()) {
    x ~> m;
  }
  x <- m;
}
