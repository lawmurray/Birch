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
    return simulate_matrix_gaussian(M.value(), V.value());
  }
  
  function logpdf(x:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(x, M.value(), V.value());
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    s1:InverseWishart?;
    m1:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    m2:MatrixNormalInverseWishart?;
    r:Distribution<Real[_,_]> <- this;

    /* match a template */
    auto compare <- V.distribution();
    if compare? && (m1 <- M.graftLinearMatrixNormalInverseWishart(compare!))? {
      r <- LinearMatrixNormalInverseWishartMatrixGaussian(m1!.A, m1!.X, m1!.C);
    } else if compare? && (m2 <- M.graftMatrixNormalInverseWishart(compare!))? {
      r <- MatrixNormalInverseWishartMatrixGaussian(m2!);
    } else if (s1 <- V.graftInverseWishart())? {
      r <- MatrixNormalInverseWishart(M, box(identity(M.rows())), s1!);
    }

    return r;
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return Gaussian(M, box(identity(M.rows())), V);
  }

  function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    prune();
    s1:InverseWishart?;
    r:MatrixNormalInverseWishart?;
    
    /* match a template */
    if (s1 <- V.graftInverseWishart())? && s1! == compare {
      r <- MatrixNormalInverseWishart(M, box(identity(M.rows())), s1!);
    }

    return r;
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
