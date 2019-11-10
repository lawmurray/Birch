/*
 * Delayed matrix Gaussian variate with matrix-normal-inverse-Wishart prior.
 */
final class DelayMatrixNormalInverseWishartMatrixGaussian(future:Real[_,_]?,
    futureUpdate:Boolean, M:DelayMatrixNormalInverseWishart) <
    DelayValue<Real[_,_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  M:DelayMatrixNormalInverseWishart& <- M;

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_wishart_matrix_gaussian(
        M.N, M.Λ, M.V.Ψ, M.V.k);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N, M.Λ, M.V.Ψ, M.V.k);
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- update_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N, M.Λ, M.V.Ψ, M.V.k);
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- downdate_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N, M.Λ, M.V.Ψ, M.V.k);
  }
}

function DelayMatrixNormalInverseWishartMatrixGaussian(
    future:Real[_,_]?, futureUpdate:Boolean,
    M:DelayMatrixNormalInverseWishart) ->
    DelayMatrixNormalInverseWishartMatrixGaussian {
  m:DelayMatrixNormalInverseWishartMatrixGaussian(future, futureUpdate, M);
  return m;
}
