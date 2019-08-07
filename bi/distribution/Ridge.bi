/**
 * Opaque distribution over parameters for Bayesian ridge regression. See
 * `Regression` for synopsis.
 *
 * Internally, it keeps parameters for a ridge regression with multiple
 * outputs. Only a single covariance parameter is necessary despite
 * multiple outputs, providing significant computational gains over multiple
 * ridge regressions with a single output.
 */
final class Ridge(M:Expression<Real[_,_]>, Σ:Expression<Real[_,_]>,
    α:Expression<Real>, β:Expression<Real[_]>) <
    Distribution<(Real[_,_],Real[_])> {
  /**
   * Weight means. Each column gives the mean of a weight vector.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Weight covariance scale.
   */
  Σ:Expression<Real[_,_]> <- Σ;

  /**
   * Common weight and likelihood covariance shape.
   */
  α:Expression<Real> <- α;
  
  /**
   * Weight and likelihood covariance scales.
   */
  β:Expression<Real[_]> <- β;
    
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {  // always graft, even when not forced
      delay <- DelayRidge(future, futureUpdate, M, Σ, α, β);
    }
  }

  function graftRidge() -> DelayRidge? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayRidge(future, futureUpdate, M, Σ, α, β);
    }
    return DelayRidge?(delay);
  }
}

/**
 * Create ridge prior.
 */
function Ridge(M:Expression<Real[_,_]>, Σ:Expression<Real[_,_]>,
    α:Expression<Real>, β:Expression<Real[_]>) -> Ridge {
  m:Ridge(M, Σ, α, β);
  return m;
}

/**
 * Create ridge prior.
 */
function Ridge(M:Real[_,_], Σ:Real[_,_], α:Real, β:Real[_]) -> Ridge {
  return Ridge(Boxed(M), Boxed(Σ), Boxed(α), Boxed(β));
}
