/**
 * Matrix Gaussian distribution where each element is independent.
 */
final class IndependentMatrixGaussian(M:Expression<Real[_,_]>,
    v:Expression<Real[_]>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;

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
      m1:TransformLinearMatrix<MatrixNormalInverseGamma>?;
      m2:MatrixNormalInverseGamma?;

      if (m1 <- M.graftLinearMatrixNormalInverseGamma())? && m1!.X.σ2 == σ2.get() {
        delay <- LinearMatrixNormalInverseGammaMatrixGaussian(future, futureUpdate, m1!.A, m1!.X, m1!.C);
      } else if (m2 <- M.graftMatrixNormalInverseGamma())? && m2!.σ2 == σ2.get() {
        delay <- MatrixNormalInverseGammaMatrixGaussian(future, futureUpdate, m2!);
      } else if (s1 <- σ2.graftIndependentInverseGamma())? {
        delay <- MatrixNormalInverseGamma(future, futureUpdate, M, identity(M.rows()), s1!);
      } else {
        delay <- MatrixGaussian(future, futureUpdate, M, identity(M.rows()), diagonal(σ2));
      }
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- MatrixGaussian(future, futureUpdate, M,
          identity(M.rows()), diagonal(σ2.value()));
    }
    return MatrixGaussian?(delay);
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:IndependentInverseGamma?;
      if (s1 <- σ2.graftIndependentInverseGamma())? {
        delay <- MatrixNormalInverseGamma(future, futureUpdate, M,
            identity(M.rows()), s1!);
      }
    }
    return MatrixNormalInverseGamma?(delay);
  }
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, σ2:Expression<Real[_]>) ->
    IndependentMatrixGaussian {
  m:IndependentMatrixGaussian(M, σ2);
  return m;
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Expression<Real[_,_]>, σ2:Real[_]) ->
    IndependentMatrixGaussian {
  return Gaussian(M, Boxed(σ2));
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Real[_,_], σ2:Expression<Real[_]>) ->
    IndependentMatrixGaussian {
  return Gaussian(Boxed(M), σ2);
}

/**
 * Create matrix Gaussian distribution where each element is independent.
 */
function Gaussian(M:Real[_,_], σ2:Real[_]) -> IndependentMatrixGaussian {
  return Gaussian(Boxed(M), Boxed(σ2));
}
