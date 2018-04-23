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
   * Updated first shape.
   */
  α_p:Real;

  /**
   * Updated second shape.
   */
  β_p:Real;

  function isBeta() -> Boolean {
    return isMissing();
  }

  function getBeta() -> (Real, Real) {
    return (α_p, β_p);
  }

  function setBeta(θ:(Real, Real)) {
    (α_p, β_p) <- θ;
  }

  function doMarginalize() {
    α_p <- α.value();
    β_p <- β.value();
  }

  function doSimulate() -> Real {
    return simulate_beta(α_p, β_p);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_beta(x, α_p, β_p);
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
