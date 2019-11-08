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
  
  function valueForward() -> Real[_,_] {
    assert !delay?;
    return simulate_matrix_normal_inverse_gamma(M, llt(Σ), σ2.α, σ2.β);
  }

  function observeForward(X:Real[_,_]) -> Real {
    assert !delay?;
    return logpdf_matrix_normal_inverse_gamma(X, M, llt(Σ), σ2.α, σ2.β);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayMatrixNormalInverseGamma(future, futureUpdate, M,
          Σ, σ2.graftIndependentInverseGamma()!);
    }
  }
}
