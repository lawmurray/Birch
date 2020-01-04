/*
 * ed Beta random variate.
 */
final class Beta(future:Real?, futureUpdate:Boolean, α:Expression<Real>,
    β:Expression<Real>) < Distribution<Real>(future, futureUpdate) {
  /**
   * First shape.
   */
  α:Expression<Real> <- α;

  /**
   * Second shape.
   */
  β:Expression<Real> <- β;

  function simulate() -> Real {
    return simulate_beta(α, β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_beta(x, α, β);
  }

  function cdf(x:Real) -> Real? {
    return cdf_beta(x, α, β);
  }

  function quantile(p:Real) -> Real? {
    return quantile_beta(p, α, β);
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function upper() -> Real? {
    return 1.0;
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- Beta(future, futureUpdate, α, β);
    }
  }

  function graftBeta() -> Beta? {
    if delay? {
      delay!.prune();
    } else {
      delay <- Beta(future, futureUpdate, α, β);
    }
    return Beta?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Beta");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

function Beta(future:Real?, futureUpdate:Boolean, α:Real, β:Real) -> Beta {
  m:Beta(future, futureUpdate, α, β);
  return m;
}

/**
 * Create beta distribution.
 */
function Beta(α:Expression<Real>, β:Expression<Real>) -> Beta {
  m:Beta(α, β);
  return m;
}

/**
 * Create beta distribution.
 */
function Beta(α:Expression<Real>, β:Real) -> Beta {
  return Beta(α, Boxed(β));
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Expression<Real>) -> Beta {
  return Beta(Boxed(α), β);
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Real) -> Beta {
  return Beta(Boxed(α), Boxed(β));
}
