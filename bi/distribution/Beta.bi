/*
 * ed Beta random variate.
 */
final class Beta(α:Expression<Real>,
    β:Expression<Real>) < Distribution<Real> {
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

  function quantile(P:Real) -> Real? {
    return quantile_beta(P, α, β);
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function upper() -> Real? {
    return 1.0;
  }

  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function graftBeta() -> Beta? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Beta");
    buffer.set("α", α);
    buffer.set("β", β);
  }
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
