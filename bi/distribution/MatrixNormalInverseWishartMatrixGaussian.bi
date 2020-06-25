/**
 * Matrix normal-inverse-Wishart-Gaussian distribution.
 */
final class MatrixNormalInverseWishartMatrixGaussian(
    M:MatrixNormalInverseWishart) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:MatrixNormalInverseWishart& <- M;

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
    auto V <- M.V;
    return simulate_matrix_normal_inverse_wishart_matrix_gaussian(
        M.N.value(), M.Λ.value(), V.Ψ.value(), V.k.value());
  }

  function simulateLazy() -> Real[_,_]? {
    auto M <- this.M;
    auto V <- M.V;
    return simulate_matrix_normal_inverse_wishart_matrix_gaussian(
        M.N.get(), M.Λ.get(), V.Ψ.get(), V.k.get());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    auto M <- this.M;
    auto V <- M.V;
    return logpdf_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), V.Ψ.value(), V.k.value());
  }

  function logpdfLazy(X:Expression<Real[_,_]>) -> Expression<Real>? {
    auto M <- this.M;
    auto V <- M.V;
    return logpdf_lazy_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N, M.Λ, V.Ψ, V.k);
  }

  function update(X:Real[_,_]) {
    auto M <- this.M;
    auto V <- M.V;
    (M.N, M.Λ, V.Ψ, V.k) <- box(update_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), V.Ψ.value(), V.k.value()));
  }

  function updateLazy(X:Expression<Real[_,_]>) {
    auto M <- this.M;
    auto V <- M.V;
    (M.N, M.Λ, V.Ψ, V.k) <- update_lazy_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N, M.Λ, V.Ψ, V.k);
  }

  function downdate(X:Real[_,_]) {
    auto M <- this.M;
    auto V <- M.V;
    (M.N, M.Λ, V.Ψ, V.k) <- box(downdate_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), V.Ψ.value(), V.k.value()));
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

function MatrixNormalInverseWishartMatrixGaussian(
    M:MatrixNormalInverseWishart) ->
    MatrixNormalInverseWishartMatrixGaussian {
  m:MatrixNormalInverseWishartMatrixGaussian(M);
  m.link();
  return m;
}
