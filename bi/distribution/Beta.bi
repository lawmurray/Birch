/**
 * Beta distribution.
 */
final class Beta(α:Expression<Real>, β:Expression<Real>) < Distribution<Real> {
  /**
   * First shape.
   */
  α:Expression<Real> <- α;

  /**
   * Second shape.
   */
  β:Expression<Real> <- β;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayBeta(future, futureUpdate, α, β);
    }
  }

  function graftBeta(child:Delay?) -> DelayBeta? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayBeta(future, futureUpdate, α, β);
    }
    return DelayBeta?(delay);
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
