/*
 * Delayed inverse-gamma random variate.
 */
final class DelayIndependentInverseGamma(future:Real[_]?,
    futureUpdate:Boolean, α:Real, β:Real[_]) < DelayValue<Real[_]>(future,
    futureUpdate) {
  /**
   * Shape.
   */
  α:Real <- α;

  /**
   * Scale.
   */
  β:Real[_] <- β;

  function simulate() -> Real[_] {
    return transform<Real>(β, @(b:Real) -> Real {
        return simulate_inverse_gamma(α, b); });
  }
  
  function logpdf(x:Real[_]) -> Real {
    return transform_reduce<Real>(x, β, 0.0, @(a:Real, b:Real) -> Real {
        return a + b;
      }, @(x:Real, b:Real) -> Real {
        return logpdf_inverse_gamma(x, α, b);
      });
  }

  function update(x:Real[_]) {
    //
  }

  function downdate(x:Real[_]) {
    //
  }

  function pdf(x:Real[_]) -> Real {
    return transform_reduce<Real>(x, β, 1.0, @(a:Real, b:Real) -> Real {
        return a*b;
      }, @(x:Real, b:Real) -> Real {
        return pdf_inverse_gamma(x, α, b);
      });
  }

  function lower() -> Real[_]? {
    return vector(0.0, length(β));
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentInverseGamma");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

function DelayIndependentInverseGamma(future:Real[_]?, futureUpdate:Boolean,
    α:Real, β:Real[_]) -> DelayIndependentInverseGamma {
  m:DelayIndependentInverseGamma(future, futureUpdate, α, β);
  return m;
}
