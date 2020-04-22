/*
 * Grafted matrix Gaussian variate with linear transformation of
 * matrix-normal-inverse-gamma prior.
 */
final class LinearMatrixNormalInverseGammaMatrixGaussian(
    A:Expression<Real[_,_]>, M:MatrixNormalInverseGamma,
    C:Expression<Real[_,_]>) < Distribution<Real[_,_]> {
  /**
   * Scale.
   */
  A:Expression<Real[_,_]> <- A;

  /**
   * Mean.
   */
  M:MatrixNormalInverseGamma <- M;

  /**
   * Offset.
   */
  C:Expression<Real[_,_]> <- C;

  function simulate() -> Real[_,_] {
    return simulate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value());
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- update_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value());
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- downdate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value());
  }

  function graftFinalize() -> Boolean {
    A.value();
    C.value();
    if !M.isRealized() {
      link();
      return true;
    } else {
      return false;
    }
  }

  function link() {
    M.setChild(this);
  }
  
  function unlink() {
    M.releaseChild();
  }
}

function LinearMatrixNormalInverseGammaMatrixGaussian(A:Expression<Real[_,_]>,
    M:MatrixNormalInverseGamma, C:Expression<Real[_,_]>) ->
    LinearMatrixNormalInverseGammaMatrixGaussian {
  m:LinearMatrixNormalInverseGammaMatrixGaussian(A, M, C);
  return m;
}
