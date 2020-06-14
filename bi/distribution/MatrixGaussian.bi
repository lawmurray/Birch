/**
 * Matrix Gaussian distribution.
 *
 * !!! note
 *     See Gaussian for associated factory functions for the creation of
 *     MatrixGaussian objects.
 */
class MatrixGaussian(M:Expression<Real[_,_]>, U:Expression<LLT>,
    V:Expression<LLT>) < Distribution<Real[_,_]> {
  /**
   * Mean.
   */
  M:Expression<Real[_,_]> <- M;
  
  /**
   * Among-row covariance.
   */
  U:Expression<LLT> <- U;

  /**
   * Among-column covariance.
   */
  V:Expression<LLT> <- V;

  function rows() -> Integer {
    return M.rows();
  }

  function columns() -> Integer {
    return M.columns();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_gaussian(M.value(), U.value(), V.value());
  }

  function simulateLazy() -> Real[_,_]? {
    return simulate_matrix_gaussian(M.get(), U.get(), V.get());
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(X, M.value(), U.value(), V.value());
  }

  function logpdfLazy(X:Expression<Real[_,_]>) -> Expression<Real>? {
    return logpdf_lazy_matrix_gaussian(X, M, U, V);
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

  function graftMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
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
function Gaussian(M:Expression<Real[_,_]>, U:Expression<LLT>,
    V:Expression<LLT>) -> MatrixGaussian {
  m:MatrixGaussian(M, U, V);
  return m;
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<LLT>,
    V:LLT) -> MatrixGaussian {
  return Gaussian(M, U, box(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:LLT, V:Expression<LLT>) ->
    MatrixGaussian {
  return Gaussian(M, box(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:LLT, V:LLT) ->
      MatrixGaussian {
  return Gaussian(M, box(U), box(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<LLT>, V:Expression<LLT>) ->
    MatrixGaussian {
  return Gaussian(box(M), U, V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<LLT>, V:LLT) -> MatrixGaussian {
  return Gaussian(box(M), U, box(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:LLT, V:Expression<LLT>) -> MatrixGaussian {
  return Gaussian(box(M), box(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:LLT, V:LLT) -> MatrixGaussian {
  return Gaussian(box(M), box(U), box(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<LLT>) -> MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:LLT) -> MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_], V:Expression<LLT>) ->
    MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_], V:LLT) ->
      MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>, V:Expression<LLT>) ->
    MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>, V:LLT) -> MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], V:Expression<LLT>) -> MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], V:LLT) -> MatrixGaussian {
  return Gaussian(M, llt(U), V);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<LLT>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<LLT>,
    V:Real[_,_]) -> MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:LLT, V:Expression<Real[_,_]>) ->
    MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:LLT, V:Real[_,_]) ->
      MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<LLT>, V:Expression<Real[_,_]>) ->
    MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<LLT>, V:Real[_,_]) ->
    MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:LLT, V:Expression<Real[_,_]>) ->
    MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:LLT, V:Real[_,_]) -> MatrixGaussian {
  return Gaussian(M, U, llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Expression<Real[_,_]>,
    V:Real[_,_]) -> MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_],
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Expression<Real[_,_]>, U:Real[_,_], V:Real[_,_]) ->
      MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>,
    V:Expression<Real[_,_]>) -> MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Expression<Real[_,_]>, V:Real[_,_]) ->
    MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], V:Expression<Real[_,_]>) ->
    MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(M:Real[_,_], U:Real[_,_], V:Real[_,_]) -> MatrixGaussian {
  return Gaussian(M, llt(U), llt(V));
}
