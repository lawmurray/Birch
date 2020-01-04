/**
 * Matrix Gaussian distribution where each column is independent.
 */
final class IndependentColumnMatrixGaussian(future:Real[_,_]?,
    futureUpdate:Boolean, M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    v:Expression<Real[_]>) < Distribution<Real[_,_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Among-row covariance.
   */
  U:Expression<Real[_,_]> <- U;

  /**
   * Among-column variances.
   */
  σ2:Expression<Real[_]> <- v;

  function rows() -> Integer {
    return M.rows();
  }

  function columns() -> Integer {
    return M.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_gaussian(M, U, σ2);
  }
  
  function logpdf(x:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(x, M, U, σ2);
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    s1:IndependentInverseGamma?;
    if (s1 <- σ2.graftIndependentInverseGamma())? {
      return MatrixNormalInverseGamma(future, futureUpdate, M, U, s1!);
    } else {
      return MatrixGaussian(future, futureUpdate, M, U, Boxed(diagonal(σ2)));
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return MatrixGaussian(future, futureUpdate, M, U, Boxed(diagonal(σ2)));
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    prune();
    s1:IndependentInverseGamma?;
    if (s1 <- σ2.graftIndependentInverseGamma())? {
      return MatrixNormalInverseGamma(future, futureUpdate, M, U, s1!);
    }
    return nil;
  }
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    σ2:Expression<Real[_]>) -> IndependentColumnMatrixGaussian {
  m:IndependentColumnMatrixGaussian(nil, true, M, U, σ2);
  return m;
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    σ2:Real[_]) -> IndependentColumnMatrixGaussian {
  return Gaussian(M, U, Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_],
    σ2:Expression<Real[_]>) -> IndependentColumnMatrixGaussian {
  return Gaussian(M, Boxed(U), σ2);
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_], σ2:Real[_]) ->
      IndependentColumnMatrixGaussian {
  return Gaussian(M, Boxed(U), Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>,
    σ2:Expression<Real[_]>) -> IndependentColumnMatrixGaussian {
  return Gaussian(Boxed(M), U, σ2);
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>, σ2:Real[_]) ->
    IndependentColumnMatrixGaussian {
  return Gaussian(Boxed(M), U, Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], σ2:Expression<Real[_]>) ->
    IndependentColumnMatrixGaussian {
  return Gaussian(Boxed(M), Boxed(U), σ2);
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], σ2:Real[_]) ->
    IndependentColumnMatrixGaussian {
  return Gaussian(Boxed(M), Boxed(U), Boxed(σ2));
}
