/**
 * Matrix normal-inverse-Wishart-Gaussian distribution.
 */
final class MatrixNormalInverseWishartMatrixGaussian(
    M:MatrixNormalInverseWishart) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:MatrixNormalInverseWishart <- M;

  function rows() -> Integer {
    return M.rows();
  }
  
  function columns() -> Integer {
    return M.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_normal_inverse_wishart_matrix_gaussian(
        M.N.value(), M.Λ.value(), M.V.Ψ.value(), M.V.k.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.V.Ψ.value(), M.V.k.value());
  }

  function update(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- box(update_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.V.Ψ.value(), M.V.k.value()));
  }

  function downdate(X:Real[_,_]) {
    (M.N, M.Λ, M.V.Ψ, M.V.k) <- box(downdate_matrix_normal_inverse_wishart_matrix_gaussian(
        X, M.N.value(), M.Λ.value(), M.V.Ψ.value(), M.V.k.value()));
  }

  function link() {
    M.setChild(this);
  }
  
  function unlink() {
    M.releaseChild();
  }
}

function MatrixNormalInverseWishartMatrixGaussian(
    M:MatrixNormalInverseWishart) ->
    MatrixNormalInverseWishartMatrixGaussian {
  m:MatrixNormalInverseWishartMatrixGaussian(M);
  m.link();
  return m;
}
