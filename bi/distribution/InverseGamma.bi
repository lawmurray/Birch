/*
 * ed inverse-gamma random variate.
 */
final class InverseGamma(future:Real?, futureUpdate:Boolean, α:Expression<Real>, β:Expression<Real>) < Distribution<Real>(future, futureUpdate) {
  /**
   * Shape.
   */
  α:Expression<Real> <- α;
  
  /**
   * Scale.
   */
  β:Expression<Real> <- β;

  function simulate() -> Real {
    return simulate_inverse_gamma(α, β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma(x, α, β);
  }

  function cdf(x:Real) -> Real? {
    return cdf_inverse_gamma(x, α, β);
  }

  function quantile(p:Real) -> Real? {
    return quantile_inverse_gamma(p, α, β);
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- InverseGamma(future, futureUpdate, α, β);
    }
  }

  function graftInverseGamma() -> InverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      delay <- InverseGamma(future, futureUpdate, α, β);
    }
    return InverseGamma?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "InverseGamma");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

function InverseGamma(future:Real?, futureUpdate:Boolean, α:Real,
    β:Real) -> InverseGamma {
  m:InverseGamma(future, futureUpdate, α, β);
  return m;
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
