/**
 * Inverse-gamma distribution.
 */
final class InverseGamma(α:Expression<Real>, β:Expression<Real>) < Distribution<Real> {
  /**
   * Shape.
   */
  α:Expression<Real> <- α;
  
  /**
   * Scale.
   */
  β:Expression<Real> <- β;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayInverseGamma(future, futureUpdate, α, β);
    }
  }

  function graftInverseGamma(child:Delay?) -> DelayInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayInverseGamma(future, futureUpdate, α, β);
    }
    return DelayInverseGamma?(delay);
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
  return InverseGamma(α, Boxed(β));
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Expression<Real>) -> InverseGamma {
  return InverseGamma(Boxed(α), β);
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Real) -> InverseGamma {
  return InverseGamma(Boxed(α), Boxed(β));
}
