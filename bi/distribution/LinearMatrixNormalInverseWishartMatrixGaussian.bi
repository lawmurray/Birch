/*
 * ed matrix Gaussian variate with linear transformation of
 * matrix-normal-inverse-Wishart prior.
 */
final class LinearMatrixNormalInverseWishartMatrixGaussian(A:Real[_,_],
    M:MatrixNormalInverseWishart, C:Real[_,_]) < Distribution<Real[_,_]> {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Mean.
   */
  M:MatrixNormalInverseWishart& <- M;

  /**
   * Offset.
   */
  C:Real[_,_] <- C;

  function simulate() -> Real[_,_] {
    return simulate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        A, M.N, M.Λ, C, M.V.Ψ, M.V.k);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.V.Ψ, M.V.k);
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- update_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.V.Ψ, M.V.k);
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- downdate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.V.Ψ, M.V.k);
  }
}

function LinearMatrixNormalInverseWishartMatrixGaussian(A:Real[_,_],
    M:MatrixNormalInverseWishart, C:Real[_,_]) ->
    LinearMatrixNormalInverseWishartMatrixGaussian {
  m:LinearMatrixNormalInverseWishartMatrixGaussian(A, M, C);
  return m;
}
