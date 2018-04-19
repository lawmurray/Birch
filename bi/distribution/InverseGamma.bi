/**
 * Inverse-gamma distribution.
 */
class InverseGamma<Type1,Type2>(α:Type1, β:Type2) < Random<Real> {
  /**
   * Shape.
   */
  α:Type1 <- α;
  
  /**
   * Scale.
   */
  β:Type2 <- β;

  function update(α:Type1, β:Type2) {
    this.α <- α;
    this.β <- β;
  }

  function doSimulate() -> Real {
    return simulate_inverse_gamma(global.value(α), global.value(β));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_inverse_gamma(x, global.value(α), global.value(β));
  }
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Real) ->
    InverseGamma<Real,Real> {
  m:InverseGamma<Real,Real>(α, β);
  m.initialize();
  return m;
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Real) ->
    InverseGamma<Expression<Real>,Real> {
  m:InverseGamma<Expression<Real>,Real>(α, β);
  m.initialize();
  return m;
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Expression<Real>) ->
    InverseGamma<Real,Expression<Real>> {
  m:InverseGamma<Real,Expression<Real>>(α, β);
  m.initialize();
  return m;
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Expression<Real>, β:Expression<Real>) ->
    InverseGamma<Expression<Real>,Expression<Real>> {
  m:InverseGamma<Expression<Real>,Expression<Real>>(α, β);
  m.initialize();
  return m;
}
