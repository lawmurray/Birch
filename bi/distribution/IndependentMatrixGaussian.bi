/*
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
    r:Distribution<Real[_,_]> <- this;

    /* match a template */
    auto compare <- σ2.distribution();
    if compare? && (m1 <- M.graftLinearMatrixNormalInverseGamma(compare!))? {
      r <- LinearMatrixNormalInverseGammaMatrixGaussian(m1!.A, m1!.X, m1!.C);
    } else if compare? && (m2 <- M.graftMatrixNormalInverseGamma(compare!))? {
      r <- MatrixNormalInverseGammaMatrixGaussian(m2!);
    } else if (s1 <- σ2.graftIndependentInverseGamma())? {
      r <- MatrixNormalInverseGamma(M, Identity(M.rows()), s1!);
    }
    
    return r;
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    prune();
    return Gaussian(M, Identity(M.rows()), diagonal(σ2));
  }

  function graftMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      MatrixNormalInverseGamma? {
    prune();
    s1:IndependentInverseGamma?;
    r:MatrixNormalInverseGamma?;
    
    /* match a template */
    if (s1 <- σ2.graftIndependentInverseGamma())? && s1! == compare {
      r <- MatrixNormalInverseGamma(M, Identity(M.rows()), s1!);
    }

    return r;
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
