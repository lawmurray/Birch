/**
 * Beta distribution.
 */
class Beta(α:Expression<Real>, β:Expression<Real>) < Random<Real> {
  /**
   * First shape parameter.
   */
  α:Expression<Real> <- α;

  /**
   * Second shape parameter.
   */
  β:Expression<Real> <- β;

  function doSimulate() -> Real {
    return simulate_beta(α.value(), β.value());
  }
  
  function doObserve(x:Real) -> Real {
    return observe_beta(x, α.value(), β.value());
  }
}

/**
 * Create beta distribution.
 */
function Beta(α:Expression<Real>, β:Expression<Real>) -> Beta {
  m:Beta(α, β);
  m.initialize();
  return m;
}

/**
 * Create beta distribution.
 */
function Beta(α:Expression<Real>, β:Real) -> Beta {
  return Beta(α, Literal(β));
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Expression<Real>) -> Beta {
  return Beta(Literal(α), β);
}

/**
 * Create beta distribution.
 */
function Beta(α:Real, β:Real) -> Beta {
  return Beta(Literal(α), Literal(β));
}
