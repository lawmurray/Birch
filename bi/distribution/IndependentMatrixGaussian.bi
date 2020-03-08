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

  function simulate() -> Real[_,_] {
    return simulate_matrix_gaussian(M.value(), σ2.value());
  }
  
  function logpdf(x:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(x, M.value(), σ2.value());
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    s1:IndependentInverseGamma?;
    m1:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    m2:MatrixNormalInverseGamma?;

    if (m1 <- M.graftLinearMatrixNormalInverseGamma())? && m1!.X.σ2 == σ2.distribution() {
      return LinearMatrixNormalInverseGammaMatrixGaussian(m1!.A, m1!.X, m1!.C);
    } else if (m2 <- M.graftMatrixNormalInverseGamma())? && m2!.σ2 == σ2.distribution() {
      return MatrixNormalInverseGammaMatrixGaussian(m2!);
    } else if (s1 <- σ2.graftIndependentInverseGamma())? {
      return MatrixNormalInverseGamma(M, Identity(M.rows()), s1!);
    } else {
      return Gaussian(M, identity(M.rows()), diagonal(σ2));
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return Gaussian(M, identity(M.rows()), diagonal(σ2));
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    prune();
    s1:IndependentInverseGamma?;
    if (s1 <- σ2.graftIndependentInverseGamma())? {
      return MatrixNormalInverseGamma(M, Identity(M.rows()), s1!);
    }
    return nil;
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
