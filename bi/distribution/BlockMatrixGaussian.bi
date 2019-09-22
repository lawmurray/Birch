/**
 * Matrix Gaussian distribution where each column is independent.
 */
final class BlockMatrixGaussian(M:Expression<Real[_,_]>,
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
  
  function valueForward() -> Real[_,_] {
    assert !delay?;
    return simulate_matrix_gaussian(M, U, σ2);
  }

  function observeForward(X:Real[_,_]) -> Real {
    assert !delay?;
    return logpdf_matrix_gaussian(X, M, U, σ2);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayIndependentInverseGamma?;
      if (s1 <- σ2.graftIndependentInverseGamma())? {
        delay <- DelayMatrixNormalInverseGamma(future, futureUpdate, M, U,
            s1!);
      } else if force {
        delay <- DelayMatrixGaussian(future, futureUpdate, M, U,
            diagonal(σ2));
      }
    }
  }

  function graftMatrixGaussian() -> DelayMatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMatrixGaussian(future, futureUpdate, M, U,
          diagonal(σ2.value()));
    }
    return DelayMatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseGamma() -> DelayMatrixNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayIndependentInverseGamma?;
      if (s1 <- σ2.graftIndependentInverseGamma())? {
        delay <- DelayMatrixNormalInverseGamma(future, futureUpdate, M, U,
            s1!);
      }
    }
    return DelayMatrixNormalInverseGamma?(delay);
  }
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    σ2:Expression<Real[_]>) -> BlockMatrixGaussian {
  m:BlockMatrixGaussian(M, U, σ2);
  return m;
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    σ2:Real[_]) -> BlockMatrixGaussian {
  return Gaussian(M, U, Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_],
    σ2:Expression<Real[_]>) -> BlockMatrixGaussian {
  return Gaussian(M, Boxed(U), σ2);
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_], σ2:Real[_]) ->
      BlockMatrixGaussian {
  return Gaussian(M, Boxed(U), Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>,
    σ2:Expression<Real[_]>) -> BlockMatrixGaussian {
  return Gaussian(Boxed(M), U, σ2);
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>, σ2:Real[_]) ->
    BlockMatrixGaussian {
  return Gaussian(Boxed(M), U, Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], σ2:Expression<Real[_]>) ->
    BlockMatrixGaussian {
  return Gaussian(Boxed(M), Boxed(U), σ2);
}

/**
 * Create matrix Gaussian distribution where each column is independent.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], σ2:Real[_]) ->
    BlockMatrixGaussian {
  return Gaussian(Boxed(M), Boxed(U), Boxed(σ2));
}
