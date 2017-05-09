import distribution.Delay;
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
   * Standard deviation (lower-triangular Cholesky factor of covariance).
   */
  L:Real[D,D];
    
  /**
   * Log-determinant of `L`.
   */
  logDetL:Real;

  function initialise(u:MultivariateGaussian) {
    super.initialise(u);
  }

  function initialise(μ:Real[_], L:Real[_,_]) {
    assert(length(μ) == D);
    assert(rows(L) == D && columns(L) == D);
    
    super.initialise();
    update(μ, L);
  }

  function update(μ:Real[_], L:Real[_,_]) {
    assert(length(μ) == D);
    assert(rows(L) == D && columns(L) == D);
    
    this.μ <- μ;
    this.L <- L;
    this.logDetL <- log(determinant(L));
  }

  /**
   * Value conversion.
   */
  function -> Real[_] {
    if (isMissing()) {
      graft();
      if (!isRealised()) {
        realise();
      }
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
      cpp{{
      x(make_view(make_index(d - 1))) = std::normal_distribution<double>(0.0, 1.0)(rng);
      }}
    }
    x <- μ + L*x;
    return x;
  }

  function observe(x:Real[_]) -> Real {
    assert(length(x) == D);
    
    return -0.5*squaredNorm(solve(L, x - μ)) - logDetL - 0.5*Real(D)*log(2.0*π);
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
function MultivariateGaussian(μ:Real[_], L:Real[_,_]) -> MultivariateGaussian {
  m:MultivariateGaussian(length(μ));
  m.initialise(μ, L);
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
  assert(length(x) == length(m));

  m.graft();
  m.realise();
  x <- m.x;
}

/**
 * Sample.
 */
function x:MultivariateGaussian <~ m:MultivariateGaussian {
  m.graft();
  m.realise();
  x.set(m.x);
}

/**
 * Observe.
 */
function x:Real[_] ~> m:MultivariateGaussian -> Real {
  assert(length(x) == length(m));

  m.graft();
  m.set(x);
  m.realise();
  return m.w;
}

/**
 * Initialise.
 */
function x:MultivariateGaussian ~ m:MultivariateGaussian {
  assert(x.isUninitialised());
  assert(length(x) == length(m));
  
  if (!x.isMissing()) {
    x ~> m;
  }
  x <- m;
}
