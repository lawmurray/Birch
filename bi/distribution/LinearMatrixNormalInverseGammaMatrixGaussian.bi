/**
 * Matrix Gaussian variate with linear transformation of
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
  M:MatrixNormalInverseGamma& <- M;

  /**
   * Offset.
   */
  C:Expression<Real[_,_]> <- C;

  function rows() -> Integer {
    return C.rows();
  }
  
  function columns() -> Integer {
    return C.columns();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_,_] {
    return simulate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value());
  }

  function simulateLazy() -> Real[_,_]? {
    return simulate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        A.get(), M.N.get(), M.Λ.get(), C.get(), M.α.get(), M.γ.get());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value());
  }

  function logpdfLazy(X:Expression<Real[_,_]>) -> Expression<Real>? {
    return logpdf_lazy_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.α, M.γ);
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- box(update_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value()));
  }

  function updateLazy(X:Expression<Real[_,_]>) {
    (M.N, M.Λ, M.α, M.γ) <- update_lazy_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A, M.N, M.Λ, C, M.α, M.γ);
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.α, M.γ) <- box(downdate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.α.value(), M.γ.value()));
  }

  function link() {
    M.setChild(this);
  }
  
  function unlink() {
    M.releaseChild(this);
  }
}

function LinearMatrixNormalInverseGammaMatrixGaussian(A:Expression<Real[_,_]>,
    M:MatrixNormalInverseGamma, C:Expression<Real[_,_]>) ->
    LinearMatrixNormalInverseGammaMatrixGaussian {
  m:LinearMatrixNormalInverseGammaMatrixGaussian(A, M, C);
  m.link();
  return m;
}
