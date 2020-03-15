/**
 * Inverse-gamma distribution.
 */
final class InverseGamma(α:Expression<Real>, β:Expression<Real>) <
    Distribution<Real> {
  /**
   * Shape.
   */
  α:Expression<Real> <- α;
  
  /**
   * Scale.
   */
  β:Expression<Real> <- β;

  function simulate() -> Real {
    return simulate_inverse_gamma(α.value(), β.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma(x, α.value(), β.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_inverse_gamma(x, α.value(), β.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_inverse_gamma(P, α.value(), β.value());
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function graftInverseGamma() -> InverseGamma? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "InverseGamma");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Expression<Real>) ->
    InverseGamma {
  m:InverseGamma(α, β);
  return m;
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Real) -> InverseGamma {
  return InverseGamma(α, Boxed(β));
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Expression<Real>) -> InverseGamma {
  return InverseGamma(Boxed(α), β);
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Real) -> InverseGamma {
  return InverseGamma(Boxed(α), Boxed(β));
}
