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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s1:InverseWishart?;
      m1:TransformLinearMatrix<MatrixNormalInverseWishart>?;
      m2:MatrixNormalInverseWishart?;

      if (m1 <- M.graftLinearMatrixNormalInverseWishart())? && m1!.X.V == V.get() {
        delay <- LinearMatrixNormalInverseWishartMatrixGaussian(future, futureUpdate, m1!.A, m1!.X, m1!.C);
      } else if (m2 <- M.graftMatrixNormalInverseWishart())? && m2!.V == V.get() {
        delay <- MatrixNormalInverseWishartMatrixGaussian(future, futureUpdate, m2!);
      } else if (s1 <- V.graftInverseWishart())? {
        delay <- MatrixNormalInverseWishart(future, futureUpdate, M, identity(M.rows()), s1!);
      } else {
        delay <- MatrixGaussian(future, futureUpdate, M, identity(M.rows()), V);
      }
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- MatrixGaussian(future, futureUpdate, M,
          identity(M.rows()), V);
    }
    return MatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    if delay? {
      delay!.prune();
    } else {
      s1:InverseWishart?;
      if (s1 <- V.graftInverseWishart())? {
        delay <- MatrixNormalInverseWishart(future, futureUpdate, M,
            identity(M.rows()), s1!);
      }
    }
    return MatrixNormalInverseWishart?(delay);
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
