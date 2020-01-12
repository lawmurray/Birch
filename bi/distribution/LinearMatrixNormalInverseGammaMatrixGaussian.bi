/*
 * ed matrix Gaussian variate with linear transformation of
 * matrix-normal-inverse-gamma prior.
 */
final class LinearMatrixNormalInverseGammaMatrixGaussian(A:Real[_,_],
    M:MatrixNormalInverseGamma, C:Real[_,_]) < Distribution<Real[_,_]> {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Mean.
   */
  M:MatrixNormalInverseGamma& <- M;

  /**
   * Offset.
   */
  C:Real[_,_] <- C;

  function simulate() -> Real[_,_] {
    return simulate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        A, M.N, M.Λ, C, M.α, M.γ);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.α, M.γ);
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- update_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.α, M.γ);
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- downdate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.α, M.γ);
  }
}

function LinearMatrixNormalInverseGammaMatrixGaussian(A:Real[_,_],
    M:MatrixNormalInverseGamma, C:Real[_,_]) ->
    LinearMatrixNormalInverseGammaMatrixGaussian {
  m:LinearMatrixNormalInverseGammaMatrixGaussian(A, M, C);
  return m;
}
