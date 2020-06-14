/**
 * Beta distribution.
 */
final class Beta(α:Expression<Real>, β:Expression<Real>) <
    Distribution<Real> {
  /**
   * First shape.
   */
  α:Expression<Real> <- α;

  /**
   * Second shape.
   */
  β:Expression<Real> <- β;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_beta(α.value(), β.value());
  }

  function simulateLazy() -> Real? {
    return simulate_beta(α.get(), β.get());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_beta(x, α.value(), β.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_beta(x, α, β);
  }

  function cdf(x:Real) -> Real? {
    return cdf_beta(x, α.value(), β.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_beta(P, α.value(), β.value());
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function upper() -> Real? {
    return 1.0;
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
  return Beta(α, box(β));
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Expression<Real>) -> Beta {
  return Beta(box(α), β);
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Real) -> Beta {
  return Beta(box(α), box(β));
}
