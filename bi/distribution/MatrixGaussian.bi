/**
 * Matrix Gaussian distribution.
 */
final class MatrixGaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Within-row covariance.
   */
  U:Expression<Real[_,_]> <- U;

  /**
   * Within-column covariance.
   */
  V:Expression<Real[_,_]> <- V;
  
  function valueForward() -> Real[_,_] {
    assert !delay?;
    return simulate_matrix_gaussian(M, U, V);
  }

  function observeForward(X:Real[_,_]) -> Real {
    assert !delay?;
    return logpdf_matrix_gaussian(X, M, U, V);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayMatrixGaussian(future, futureUpdate, M, U, V);
    }
  }

  function graftMatrixGaussian() -> DelayMatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMatrixGaussian(future, futureUpdate, M, U, V);
    }
    return DelayMatrixGaussian?(delay);
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
