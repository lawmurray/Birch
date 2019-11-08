/*
 * Delayed matrix Gaussian variate with linear transformation of
 * matrix-normal-inverse-Wishart prior.
 */
final class DelayLinearMatrixNormalInverseWishartMatrixGaussian(
    future:Real[_,_]?, futureUpdate:Boolean, A:Real[_,_],
    M:DelayMatrixNormalInverseWishart, C:Real[_,_]) < DelayValue<Real[_,_]>(
    future, futureUpdate) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;

  /**
   * Mean.
   */
  M:DelayMatrixNormalInverseWishart& <- M;

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

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayLinearMatrixNormalInverseWishartMatrixGaussian(
    future:Real[_,_]?, futureUpdate:Boolean, A:Real[_,_],
    M:DelayMatrixNormalInverseWishart, C:Real[_,_]) ->
    DelayLinearMatrixNormalInverseWishartMatrixGaussian {
  m:DelayLinearMatrixNormalInverseWishartMatrixGaussian(future, futureUpdate,
      A, M, C);
  return m;
}
