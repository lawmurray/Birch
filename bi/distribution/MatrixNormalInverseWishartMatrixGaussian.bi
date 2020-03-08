/*
 * ed matrix Gaussian variate with matrix-normal-inverse-Wishart prior.
 */
final class MatrixNormalInverseWishartMatrixGaussian(
    M:MatrixNormalInverseWishart) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:MatrixNormalInverseWishart& <- M;

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_wishart_matrix_gaussian(
        M.N.value(), M.Λ, M.V.Ψ.value(), M.V.k.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ, M.V.Ψ.value(), M.V.k.value());
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- update_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ, M.V.Ψ.value(), M.V.k.value());
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- downdate_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ, M.V.Ψ.value(), M.V.k.value());
  }
}

function MatrixNormalInverseWishartMatrixGaussian(
    M:MatrixNormalInverseWishart) ->
    MatrixNormalInverseWishartMatrixGaussian {
  m:MatrixNormalInverseWishartMatrixGaussian(M);
  return m;
}
