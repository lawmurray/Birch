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

  /**
   * Updated shape.
   */
  α_p:Real;

  /**
   * Updated scale.
   */
  β_p:Real;

  function isInverseGamma() -> Boolean {
    return isMissing();
  }

  function getInverseGamma() -> (Real, Real) {
    return (α_p, β_p);
  }

  function setInverseGamma(θ:(Real, Real)) {
    (α_p, β_p) <- θ;
  }

  function isScaledInverseGamma(σ2:Expression<Real>) -> Boolean {
    return isMissing() && Object(this) == Object(σ2);
  }

  function getScaledInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real) {
    assert Object(this) == Object(σ2);
    return (1.0, α_p, β_p);
  }

  function setScaledInverseGamma(σ2:Expression<Real>, θ:(Real, Real)) {
    assert Object(this) == Object(σ2);
    (α_p, β_p) <- θ;
  }

  function doMarginalize() {
    α_p <- α.value();
    β_p <- β.value();
  }

  function doSimulate() -> Real {
    return simulate_inverse_gamma(α_p, β_p);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_inverse_gamma(x, α_p, β_p);
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
