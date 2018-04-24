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
  
  /**
   * Delayed sampling node.
   */
  delay:DelayBeta?;

  function isBeta() -> Boolean {
    return isMissing();
  }

  function getBeta() -> DelayBeta {
    if (!delay?) {
      delay:DelayBeta(this, α.value(), β.value());
      this.delay <- delay;
    }
    return delay!;
  }

  function doSimulate() -> Real {
    return delay!.doSimulate();
  }
  
  function doObserve(x:Real) -> Real {
    return delay!.doObserve(x);
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
