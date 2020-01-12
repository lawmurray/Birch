/**
 * Matrix Gaussian distribution where each row is independent.
 */
final class IndependentRowMatrixGaussian(M:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Among-column covariance.
   */
  V:Expression<Real[_,_]> <- V;

  function rows() -> Integer {
    return M.rows();
  }

  function columns() -> Integer {
    return M.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_gaussian(M, V);
  }
  
  function logpdf(x:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(x, M, V);
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    s1:InverseWishart?;
    m1:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    m2:MatrixNormalInverseWishart?;

    if (m1 <- M.graftLinearMatrixNormalInverseWishart())? && m1!.X.V == V.distribution() {
      return LinearMatrixNormalInverseWishartMatrixGaussian(m1!.A, m1!.X, m1!.C);
    } else if (m2 <- M.graftMatrixNormalInverseWishart())? && m2!.V == V.distribution() {
      return MatrixNormalInverseWishartMatrixGaussian(m2!);
    } else if (s1 <- V.graftInverseWishart())? {
      return MatrixNormalInverseWishart(M, identity(M.rows()), s1!);
    } else {
      return Gaussian(M, Boxed(identity(M.rows())), V);
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return Gaussian(M, Boxed(identity(M.rows())), V);
  }

  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    prune();
    s1:InverseWishart?;
    if (s1 <- V.graftInverseWishart())? {
      return MatrixNormalInverseWishart(M, identity(M.rows()), s1!);
    }
    return nil;
  }
}

/**
 * Create matrix Gaussian distribution where each row is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, V:Expression<Real[_,_]>) ->
    IndependentRowMatrixGaussian {
  m:IndependentRowMatrixGaussian(M, V);
  return m;
}

/**
 * Create matrix Gaussian distribution where each row is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, V:Real[_,_]) ->
    IndependentRowMatrixGaussian {
  return Gaussian(M, Boxed(V));
}

/**
 * Create matrix Gaussian distribution where each row is independent.
 */
function Gaussian(M:Real[_,_], V:Expression<Real[_,_]>) ->
    IndependentRowMatrixGaussian {
  return Gaussian(Boxed(M), V);
}
