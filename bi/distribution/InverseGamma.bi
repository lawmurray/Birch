/**
 * Inverse-gamma distribution.
 */
class InverseGamma(α:Expression<Real>, β:Expression<Real>) < Random<Real> {
  /**
   * Shape.
   */
  α:Expression<Real> <- α;
  
  /**
   * Scale.
   */
  β:Expression<Real> <- β;

  function doSimulate() -> Real {
    return simulate_inverse_gamma(α.value(), β.value());
  }
  
  function doObserve(x:Real) -> Real {
    return observe_inverse_gamma(x, α.value(), β.value());
  }
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Expression<Real>) -> InverseGamma {
  m:InverseGamma(α, β);
  m.initialize();
  return m;
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Real) -> InverseGamma {
  return InverseGamma(α, Literal(β));
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Expression<Real>) -> InverseGamma {
  return InverseGamma(Literal(α), β);
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Real) -> InverseGamma {
  return InverseGamma(Literal(α), Literal(β));
}
