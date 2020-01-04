/**
 * Matrix Gaussian distribution where each column is independent.
 */
final class IndependentColumnMatrixGaussian(M:Expression<Real[_,_]>,
    U:Expression<Real[_,_]>, v:Expression<Real[_]>) < Distribution<Real[_,_]> {
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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s1:IndependentInverseGamma?;
      if (s1 <- σ2.graftIndependentInverseGamma())? {
        delay <- MatrixNormalInverseGamma(future, futureUpdate, M, U,
            s1!);
      } else {
        delay <- MatrixGaussian(future, futureUpdate, M, U,
            diagonal(σ2));
      }
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- MatrixGaussian(future, futureUpdate, M, U,
          diagonal(σ2.value()));
    }
    return MatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:IndependentInverseGamma?;
      if (s1 <- σ2.graftIndependentInverseGamma())? {
        delay <- MatrixNormalInverseGamma(future, futureUpdate, M, U,
            s1!);
      }
    }
    return MatrixNormalInverseGamma?(delay);
  }
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    σ2:Expression<Real[_]>) -> IndependentColumnMatrixGaussian {
  m:IndependentColumnMatrixGaussian(M, U, σ2);
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
