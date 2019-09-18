/*
 * Delayed inverse-gamma random variate.
 */
final class DelayIndependentInverseGamma(future:Real[_]?,
    futureUpdate:Boolean, α:Real[_], β:Real[_]) < DelayValue<Real[_]>(future,
    futureUpdate) {
  /**
   * Shape.
   */
  α:Real[_] <- α;

  /**
   * Scale.
   */
  β:Real[_] <- β;

  function simulate() -> Real[_] {
    return transform<Real>(α, β, @(a:Real, b:Real) -> Real {
        return simulate_inverse_gamma(a, b); });
  }
  
  function logpdf(x:Real[_]) -> Real {
    return transform_reduce<Real>(x, α, β, 0.0, @(a:Real, b:Real) -> Real {
        return a + b;
      }, @(x:Real, a:Real, b:Real) -> Real {
        return logpdf_inverse_gamma(x, a, b);
      });
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real[_]) -> Real {
    return transform_reduce<Real>(x, α, β, 1.0, @(a:Real, b:Real) -> Real {
        return a*b;
      }, @(x:Real, a:Real, b:Real) -> Real {
        return pdf_inverse_gamma(x, a, b);
      });
  }

  function lower() -> Real[_]? {
    return vector(0.0, length(α));
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentInverseGamma");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

function DelayIndependentInverseGamma(future:Real[_]?, futureUpdate:Boolean,
    α:Real[_], β:Real[_]) -> DelayIndependentInverseGamma {
  assert length(α) == length(β);
  m:DelayIndependentInverseGamma(future, futureUpdate, α, β);
  return m;
}
