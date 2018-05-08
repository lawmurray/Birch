/**
 * Beta distribution.
 */
class Beta(α:Expression<Real>, β:Expression<Real>) < Distribution<Real> {
  /**
   * First shape.
   */
  α:Expression<Real> <- α;

  /**
   * Second shape.
   */
  β:Expression<Real> <- β;

  function graft() -> DelayValue<Real> {
    return DelayBeta(α, β);
  }

  function graftBeta() -> DelayBeta? {
    return DelayBeta(α, β);
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
