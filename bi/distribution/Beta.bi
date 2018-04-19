/**
 * Beta distribution.
 */
class Beta<Type1,Type2>(α:Type1, β:Type2) < Random<Real> {
  /**
   * First shape parameter.
   */
  α:Type1 <- α;

  /**
   * Second shape parameter.
   */
  β:Type2 <- β;

  function doSimulate() -> Real {
    return simulate_beta(global.value(α), global.value(β));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_beta(x, global.value(α), global.value(β));
  }
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Real) -> Beta<Real,Real> {
  m:Beta<Real,Real>(α, β);
  m.initialize();
  return m;
}

/**
 * Create beta distribution.
 */
function Beta(α:Expression<Real>, β:Real) -> Beta<Expression<Real>,Real> {
  m:Beta<Expression<Real>,Real>(α, β);
  m.initialize();
  return m;
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Expression<Real>) -> Beta<Real,Expression<Real>> {
  m:Beta<Real,Expression<Real>>(α, β);
  m.initialize();
  return m;
}

/**
 * Create beta distribution.
 */
function Beta(α:Expression<Real>, β:Expression<Real>) ->
    Beta<Expression<Real>,Expression<Real>> {
  m:Beta<Expression<Real>,Expression<Real>>(α, β);
  m.initialize();
  return m;
}
