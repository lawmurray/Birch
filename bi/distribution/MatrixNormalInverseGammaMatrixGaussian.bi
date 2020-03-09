/*
 * ed matrix Gaussian variate with matrix-normal-inverse-gamma prior.
 */
final class MatrixNormalInverseGammaMatrixGaussian(
    M:MatrixNormalInverseGamma) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:MatrixNormalInverseGamma& <- M;

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_gamma_matrix_gaussian(
        M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- update_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- downdate_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }
}

function MatrixNormalInverseGammaMatrixGaussian(
    M:MatrixNormalInverseGamma) -> MatrixNormalInverseGammaMatrixGaussian {
  m:MatrixNormalInverseGammaMatrixGaussian(M);
  return m;
}
