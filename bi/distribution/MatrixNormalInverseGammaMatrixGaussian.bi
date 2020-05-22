/**
 * Matrix normal-inverse-gamma-Gaussian distribution.
 */
final class MatrixNormalInverseGammaMatrixGaussian(
    M:MatrixNormalInverseGamma) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:MatrixNormalInverseGamma <- M;

  function rows() -> Integer {
    return M.rows();
  }
  
  function columns() -> Integer {
    return M.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_gamma_matrix_gaussian(
        M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- box(update_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value()));
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- box(downdate_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value()));
  }

  function link() {
    M.setChild(this);
  }
  
  function unlink() {
    M.releaseChild(this);
  }
}

function MatrixNormalInverseGammaMatrixGaussian(
    M:MatrixNormalInverseGamma) -> MatrixNormalInverseGammaMatrixGaussian {
  m:MatrixNormalInverseGammaMatrixGaussian(M);
  m.link();
  return m;
}
