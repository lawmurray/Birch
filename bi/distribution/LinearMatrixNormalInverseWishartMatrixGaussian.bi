/**
 * Matrix Gaussian variate with linear transformation of
 * matrix-normal-inverse-Wishart prior.
 */
final class LinearMatrixNormalInverseWishartMatrixGaussian(
    A:Expression<Real[_,_]>, M:MatrixNormalInverseWishart,
    C:Expression<Real[_,_]>) < Distribution<Real[_,_]> {
  /**
   * Scale.
   */
  A:Expression<Real[_,_]> <- A;

  /**
   * Mean.
   */
  M:MatrixNormalInverseWishart& <- M;

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
    auto M <- this.M;
    auto V <- M.V;
    return simulate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        A.value(), M.N.value(), M.Λ.value(), C.value(), V.Ψ.value(), V.k.value());
  }

  function simulateLazy() -> Real[_,_]? {
    auto M <- this.M;
    auto V <- M.V;
    return simulate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        A.get(), M.N.get(), M.Λ.get(), C.get(), V.Ψ.get(), V.k.get());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    auto M <- this.M;
    auto V <- M.V;
    return logpdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), V.Ψ.value(), V.k.value());
  }

  function logpdfLazy(X:Expression<Real[_,_]>) -> Expression<Real>? {
    auto M <- this.M;
    auto V <- M.V;
    return logpdf_lazy_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A, M.N, M.Λ, C, V.Ψ, V.k);
  }

  function update(X:Real[_,_]) {
    auto M <- this.M;
    auto V <- M.V;
    (M.N, M.Λ, V.Ψ, V.k) <- box(update_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), V.Ψ.value(), V.k.value()));
  }

  function updateLazy(X:Expression<Real[_,_]>) {
    auto M <- this.M;
    auto V <- M.V;
    (M.N, M.Λ, V.Ψ, V.k) <- update_lazy_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A, M.N, M.Λ, C, V.Ψ, V.k);
  }

  function downdate(X:Real[_,_]) {
    auto M <- this.M;
    auto V <- M.V;
    (M.N, M.Λ, V.Ψ, V.k) <- box(downdate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), V.Ψ.value(), V.k.value()));
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

function LinearMatrixNormalInverseWishartMatrixGaussian(
    A:Expression<Real[_,_]>, M:MatrixNormalInverseWishart,
    C:Expression<Real[_,_]>) -> LinearMatrixNormalInverseWishartMatrixGaussian {
  m:LinearMatrixNormalInverseWishartMatrixGaussian(A, M, C);
  m.link();
  return m;
}
