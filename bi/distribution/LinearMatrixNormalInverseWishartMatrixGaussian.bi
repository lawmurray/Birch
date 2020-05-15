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
  M:MatrixNormalInverseWishart <- M;

  /**
   * Offset.
   */
  C:Expression<Real[_,_]> <- C;

  function simulate() -> Real[_,_] {
    return simulate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        A.value(), M.N.value(), M.Λ.value(), C.value(), M.V.Ψ.value(), M.V.k.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.V.Ψ.value(), M.V.k.value());
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- box(update_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.V.Ψ.value(), M.V.k.value()));
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- box(downdate_linear_matrix_normal_inverse_wishart_matrix_gaussian(
        X, A.value(), M.N.value(), M.Λ.value(), C.value(), M.V.Ψ.value(), M.V.k.value()));
  }

  function link() {
    M.setChild(this);
  }
  
  function unlink() {
    M.releaseChild();
  }
}

function LinearMatrixNormalInverseWishartMatrixGaussian(
    A:Expression<Real[_,_]>, M:MatrixNormalInverseWishart,
    C:Expression<Real[_,_]>) -> LinearMatrixNormalInverseWishartMatrixGaussian {
  m:LinearMatrixNormalInverseWishartMatrixGaussian(A, M, C);
  m.link();
  return m;
}
