/*
 * ed matrix Gaussian variate with matrix-normal-inverse-gamma prior.
 */
final class MatrixNormalInverseGammaMatrixGaussian(future:Real[_,_]?,
    futureUpdate:Boolean, M:MatrixNormalInverseGamma) <
    Distribution<Real[_,_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  M:MatrixNormalInverseGamma& <- M;

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_gamma_matrix_gaussian(
        M.N, M.Λ, M.α, M.γ);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N, M.Λ, M.α, M.γ);
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- update_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N, M.Λ, M.α, M.γ);
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- downdate_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N, M.Λ, M.α, M.γ);
  }
}

function MatrixNormalInverseGammaMatrixGaussian(
    future:Real[_,_]?, futureUpdate:Boolean,
    M:MatrixNormalInverseGamma) ->
    MatrixNormalInverseGammaMatrixGaussian {
  m:MatrixNormalInverseGammaMatrixGaussian(future, futureUpdate, M);
  return m;
}
