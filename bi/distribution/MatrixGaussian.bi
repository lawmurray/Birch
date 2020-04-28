/*
 * Matrix Gaussian distribution.
 */
class MatrixGaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Among-row covariance.
   */
  U:Expression<Real[_,_]> <- U;

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
    return simulate_matrix_gaussian(M.value(), U.value(), V.value());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(X, M.value(), U.value(), V.value());
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    s1:InverseWishart?;
    r:Distribution<Real[_,_]> <- this;
    
    /* match a template */
    if (s1 <- V.graftInverseWishart())? {
      r <- MatrixNormalInverseWishart(M, U, s1!);
    }
    
    return r;
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return this;
  }

  function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    prune();
    s1:InverseWishart?;
    r:MatrixNormalInverseWishart?;
    
    /* match a template */
    if (s1 <- V.graftInverseWishart())? && s1! == compare {
      r <- MatrixNormalInverseWishart(M, U, s1!);
    }

    return r;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixGaussian");
    buffer.set("M", M);
    buffer.set("U", U);
    buffer.set("V", V);
  }
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  m:MatrixGaussian(M, U, V);
  return m;
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Real[_,_]) -> MatrixGaussian {
  return Gaussian(M, U, Boxed(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_],
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return Gaussian(M, Boxed(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_], V:Real[_,_]) ->
      MatrixGaussian {
  return Gaussian(M, Boxed(U), Boxed(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return Gaussian(Boxed(M), U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>, V:Real[_,_]) ->
    MatrixGaussian {
  return Gaussian(Boxed(M), U, Boxed(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], V:Expression<Real[_,_]>) ->
    MatrixGaussian {
  return Gaussian(Boxed(M), Boxed(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], V:Real[_,_]) -> MatrixGaussian {
  return Gaussian(Boxed(M), Boxed(U), Boxed(V));
}
