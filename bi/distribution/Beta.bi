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

  function valueForward() -> Real {
    assert !delay?;
    return simulate_beta(α, β);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_beta(x, α, β);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayBeta(future, futureUpdate, α, β);
    }
  }

  function graftBeta() -> DelayBeta? {
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
