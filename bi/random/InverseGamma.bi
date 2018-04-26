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

  function isInverseGamma() -> Boolean {
    return isMissing();
  }

  function getInverseGamma() -> DelayInverseGamma {
    if (!delay?) {
      delay:DelayInverseGamma(this, α.value(), β.value());
      this.delay <- delay;
    }
    return DelayInverseGamma?(delay)!;
  }

  function isScaledInverseGamma(σ2:Expression<Real>) -> Boolean {
    return isMissing() && Object(this) == Object(σ2);
  }

  function getScaledInverseGamma(σ2:Expression<Real>) -> (Real, DelayInverseGamma) {
    if (!delay?) {
      delay:DelayInverseGamma(this, α.value(), β.value());
      this.delay <- delay;
    }
    return (1.0, DelayInverseGamma?(delay)!);
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
