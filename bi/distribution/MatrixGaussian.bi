/*
 * ed matrix Gaussian random variate.
 */
class MatrixGaussian(future:Real[_,_]?, futureUpdate:Boolean,
    M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) < Distribution<Real[_,_]>(future, futureUpdate) {
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
    return simulate_matrix_gaussian(M, U, V);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(X, M, U, V);
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    s1:InverseWishart?;
    if (s1 <- V.graftInverseWishart())? {
      return MatrixNormalInverseWishart(future, futureUpdate, M, U, s1!);
    } else {
      return this;
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return this;
  }

  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    prune();
    s1:InverseWishart?;
    if (s1 <- V.graftInverseWishart())? {
      return MatrixNormalInverseWishart(future, futureUpdate, M, U, s1!);
    }
    return nil;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixGaussian");
    buffer.set("M", M);
    buffer.set("U", U);
    buffer.set("V", V);
  }
}

function MatrixGaussian(future:Real[_,_]?, futureUpdate:Boolean,
    M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  m:MatrixGaussian(future, futureUpdate, M, U, V);
  return m;
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return MatrixGaussian(nil, true, M, U, V);
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
