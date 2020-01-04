/*
 * ed Weibull random variate.
 */
final class Weibull(future:Real?, futureUpdate:Boolean, k:Expression<Real>,
    λ:Expression<Real>) < Distribution<Real>(future, futureUpdate) {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;

  /**
   * Scale.
   */
  λ:Expression<Real> <- λ;

  function simulate() -> Real {
    return simulate_weibull(k, λ);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_weibull(x, k, λ);
  }

  function cdf(x:Real) -> Real? {
    return cdf_weibull(x, k, λ);
  }

  function quantile(p:Real) -> Real? {
    return quantile_weibull(p, k, λ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Weibull");
    buffer.set("k", k);
    buffer.set("λ", λ);
  }
}

function Weibull(future:Real?, futureUpdate:Boolean, k:Expression<Real>,
    λ:Expression<Real>) -> Weibull {
  m:Weibull(future, futureUpdate, k, λ);
  return m;
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Expression<Real>, λ:Expression<Real>) -> Weibull {
  m:Weibull(nil, true, k, λ);
  return m;
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Expression<Real>, λ:Real) -> Weibull {
  return Weibull(k, Boxed(λ));
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Real, λ:Expression<Real>) -> Weibull {
  return Weibull(Boxed(k), λ);
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Real, λ:Real) -> Weibull {
  return Weibull(Boxed(k), Boxed(λ));
}
