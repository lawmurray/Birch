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

  function graft() -> DelayValue<Real>? {
    if (delay?) {
      return delay;
    } else {
      return DelayInverseGamma(this, α, β);
    }
  }

  function graftInverseGamma() -> DelayInverseGamma? {
    if (delay?) {
      return DelayInverseGamma?(delay);
    } else {
      return DelayInverseGamma(this, α, β);
    }
  }
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Expression<Real>) -> InverseGamma {
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
