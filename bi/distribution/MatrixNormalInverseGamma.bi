/**
 * Matrix normal-inverse-gamma distribution.
 */
final class MatrixNormalInverseGamma(M:Expression<Real[_,_]>,
    Σ:Expression<Real[_,_]>, α:Expression<Real>, β:Expression<Real[_]>) <
    Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;

  /**
   * Covariance scale.
   */
  σ2:IndependentInverseGamma(α, β);

  function rows() -> Integer {
    return M.rows();
  }

  function columns() -> Integer {
    return M.columns();
  }

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMatrixNormalInverseGamma(future, futureUpdate, M,
          Σ, σ2.graftIndependentInverseGamma(child)!);
    }
  }
}
