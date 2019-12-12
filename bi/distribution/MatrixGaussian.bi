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

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseWishart?;
      if (s1 <- V.graftInverseWishart(child))? {
        delay <- DelayMatrixNormalInverseWishart(future, futureUpdate, M, U, s1!);
      } else {
        delay <- DelayMatrixGaussian(future, futureUpdate, M, U, V);
      }
    }
  }

  function graftMatrixGaussian(child:Delay?) -> DelayMatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMatrixGaussian(future, futureUpdate, M, U, V);
    }
    return DelayMatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseWishart(child:Delay?) -> DelayMatrixNormalInverseWishart? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseWishart?;
      if (s1 <- V.graftInverseWishart(child))? {
        delay <- DelayMatrixNormalInverseWishart(future, futureUpdate, M, U,
            s1!);
      }
    }
    return DelayMatrixNormalInverseWishart?(delay);
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
