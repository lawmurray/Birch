/*
 * Delayed matrix Gaussian variate with matrix-normal-inverse-gamma prior.
 */
final class DelayMatrixNormalInverseGammaMatrixGaussian(future:Real[_,_]?,
    futureUpdate:Boolean, M:DelayMatrixNormalInverseGamma) <
    DelayValue<Real[_,_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  M:DelayMatrixNormalInverseGamma& <- M;

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

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayMatrixNormalInverseGammaMatrixGaussian(
    future:Real[_,_]?, futureUpdate:Boolean,
    M:DelayMatrixNormalInverseGamma) ->
    DelayMatrixNormalInverseGammaMatrixGaussian {
  m:DelayMatrixNormalInverseGammaMatrixGaussian(future, futureUpdate, M);
  return m;
}
