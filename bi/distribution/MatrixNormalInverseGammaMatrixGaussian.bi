/**
 * Matrix normal-inverse-gamma-Gaussian distribution.
 */
final class MatrixNormalInverseGammaMatrixGaussian(
    M:MatrixNormalInverseGamma) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:MatrixNormalInverseGamma& <- M;

  function rows() -> Integer {
    auto M <- this.M;
    return M.rows();
  }
  
  function columns() -> Integer {
    auto M <- this.M;
    return M.columns();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_,_] {
    auto M <- this.M;
    return simulate_matrix_normal_inverse_gamma_matrix_gaussian(
        M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }

  function simulateLazy() -> Real[_,_]? {
    auto M <- this.M;
    return simulate_matrix_normal_inverse_gamma_matrix_gaussian(
        M.N.get(), M.Λ.get(), M.α.get(), M.γ.get());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    auto M <- this.M;
    return logpdf_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value());
  }

  function logpdfLazy(X:Expression<Real[_,_]>) -> Expression<Real>? {
    auto M <- this.M;
    return logpdf_lazy_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N, M.Λ, M.α, M.γ);
  }

  function update(X:Real[_,_]) {
    auto M <- this.M;
    (M.N, M.Λ, M.α, M.γ) <- box(update_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value()));
  }

  function updateLazy(X:Expression<Real[_,_]>) {
    auto M <- this.M;
    (M.N, M.Λ, M.α, M.γ) <- update_lazy_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N, M.Λ, M.α, M.γ);
  }

  function downdate(X:Real[_,_]) {
    auto M <- this.M;
    (M.N, M.Λ, M.α, M.γ) <- box(downdate_matrix_normal_inverse_gamma_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.α.value(), M.γ.value()));
  }

  function link() {
    auto M <- this.M;
    M.setChild(this);
  }
  
  function unlink() {
    auto M <- this.M;
    M.releaseChild(this);
  }
}

function MatrixNormalInverseGammaMatrixGaussian(M:MatrixNormalInverseGamma)
    -> MatrixNormalInverseGammaMatrixGaussian {
  m:MatrixNormalInverseGammaMatrixGaussian(M);
  m.link();
  return m;
}
