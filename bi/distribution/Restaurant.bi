/**
 * Chinese restaurant process (CRP).
 *
 * A Random assigned this distribution cannot be instantiated or its
 * likelihood evaluated. The distribution exists only for use as a prior on
 * a Categorical distribution.
 */
final class Restaurant(α:Expression<Real>, θ:Expression<Real>) <
    Distribution<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real> <- α;
  
  /**
   * Strength.
   */
  θ:Expression<Real> <- θ;

  /**
   * Number of samples drawn in each component.
   */
  n:Integer[_];

  /**
   * Number of components enumerated.
   */
  K:Integer <- 0;

  /**
   * Number of samples drawn.
   */
  N:Integer <- 0;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Real[_] {
    assert false;
    return vector(0.0, 0);
  }

  function simulateLazy() -> Real[_]? {
    assert false;
    return vector(0.0, 0);
  }
  
  function logpdf(x:Real[_]) -> Real {
    assert false;
    return 0.0;
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    assert false;
    return box(0.0);
  }

  function graftRestaurant() -> Restaurant? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Restaurant");
    buffer.set("α", α);
    buffer.set("θ", θ);
    buffer.set("n", n);
  }
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Expression<Real>, θ:Expression<Real>) -> Restaurant {
  m:Restaurant(α, θ);
  return m;
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Expression<Real>, θ:Real) -> Restaurant {
  return Restaurant(α, box(θ));
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Expression<Real>) -> Restaurant {
  return Restaurant(box(α), θ);
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Real) -> Restaurant {
  return Restaurant(box(α), box(θ));
}
