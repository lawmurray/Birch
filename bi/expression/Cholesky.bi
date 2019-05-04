/**
 * Cholesky decomposition of a symmetric positive definite matrix.
 */
final class Cholesky(S:Real[_,_]) < Expression<Real[_,_]> {  
  /**
   * The original matrix.
   */
  S:Real[_,_] <- S;
  
  /**
   * The lower-triangular Cholesky factor, once computed.
   */
  L:Real[rows(S),columns(S)];
  
  /**
   * Has the decomposition been computed?
   */
  computed:Boolean <- false;

  /*
   * Eigen internals. Unfortunately Eigen::LLT does not support in place
   * matrix decompositions using Eigen::Map, which is how Birch wraps its own
   * array buffers for use by Eigen. instead we use an Eigen::Matrix type and
   * copy `S` into it later.
   */
  hpp{{
  Eigen::LLT<libbirch::EigenMatrix<Real>> llt;
  }}

  /**
   * Lower triangular factor.
   */
  function value() -> Real[_,_] {
    compute();
    return L;
  }

  /**
   * Rank one update (or downdate) of the decomposition. If the original
   * matrix $S$ was decomposed as $S = LL^T$, this updates the matrix $L$ so
   * that $S + axx^\top = LL^T$.
   */
  function rankUpdate(x:Real[_], a:Real) {
    compute();
    cpp{{
    llt.rankUpdate(x.toEigen(), a);
    L.toEigen() = llt.matrixL();
    }}
  }
  
  /**
   * Compute the decomposition.
   */
  function compute() {
    if !computed {
      cpp{{
      llt.compute(S.toEigen());
      L.toEigen() = llt.matrixL();
      }}
      computed <- true;
    }
  }
}

/**
 * Construct a Cholesky decomposition of a matrix.
 */
function Cholesky(S:Real[_,_]) -> Cholesky {
  m:Cholesky(S);
  return m;
}
