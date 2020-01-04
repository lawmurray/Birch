/*
 * ed Chinese restaurant process (CRP) random variate. Such a variate
 * cannot be instantiated, but the associated random variable may be
 * marginalized out.
 */
final class Restaurant(future:Real[_]?, futureUpdate:Boolean, α:Expression<Real>, θ:Expression<Real>) < Distribution<Real[_]>(future, futureUpdate) {
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

  function simulate() -> Real[_] {
    assert false;
    return vector(0.0, 0);
  }
  
  function logpdf(x:Real[_]) -> Real {
    assert false;
    return 0.0;
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- Restaurant(future, futureUpdate, α, θ);
    }
  }

  function graftRestaurant() -> Restaurant? {
    if delay? {
      delay!.prune();
    } else {
      delay <- Restaurant(future, futureUpdate, α, θ);
    }
    return Restaurant?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Restaurant");
    buffer.set("α", α);
    buffer.set("θ", θ);
    buffer.set("n", n);
  }
}

function Restaurant(future:Real[_]?, futureUpdate:Boolean, α:Real,
    θ:Real) -> Restaurant {
  m:Restaurant(future, futureUpdate, α, θ);
  return m;
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Expression<Real>, θ:Expression<Real>) -> Restaurant {
  x:Restaurant(α, θ);
  return x;
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Expression<Real>, θ:Real) -> Restaurant {
  return Restaurant(α, Boxed(θ));
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Expression<Real>) -> Restaurant {
  return Restaurant(Boxed(α), θ);
}

/**
 * Create Chinese restaurant process.
 */
function Restaurant(α:Real, θ:Real) -> Restaurant {
  return Restaurant(Boxed(α), Boxed(θ));
}
