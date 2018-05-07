/**
 * Beta distribution.
 */
class Beta(α:Expression<Real>, β:Expression<Real>) < Random<Real> {
  /**
   * First shape.
   */
  α:Expression<Real> <- α;

  /**
   * Second shape.
   */
  β:Expression<Real> <- β;

  function graft() -> DelayValue<Real>? {
    if (delay?) {
      return delay;
    } else {
      return DelayBeta(this, α, β);
    }
  }

  function graftBeta() -> DelayBeta? {
    if (delay?) {
      return DelayBeta?(delay);
    } else {
      return DelayBeta(this, α, β);
    }
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
