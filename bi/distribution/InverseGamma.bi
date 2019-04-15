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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayInverseGamma(x, α, β);
    }
  }

  function graftInverseGamma() -> DelayInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayInverseGamma(x, α, β);
    }
    return DelayInverseGamma?(delay);
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "InverseGamma");
      buffer.set("α", α.value());
      buffer.set("β", β.value());
    }
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
