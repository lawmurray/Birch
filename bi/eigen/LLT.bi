/**
 * Cholesky decomposition of a symmetric positive definite matrix, $S = LL^T$.
 *
 * The object acts as the matrix $S$, defines conversion to and assignment
 * from `Real[_,_]`, and is intended as more or less a drop-in replacement
 * for that type, albeit sharing, as usual for objects (i.e. copy-by-reference
 * rather than copy-by-value semantics). That sharing permits, for example,
 * multiple multivariate Gaussian distributions to share the same covariance
 * or precision matrix with common posterior updates performed only once.
 *
 * Various functions, such as `solve`, have overloads that make use of `LLT`
 * objects for more efficient computation.
 *
 * !!! important
 *     To emphasize, the matrix represented is $S$, not $L$, which is to say,
 *     code such as the following:
 *
 *         auto A <- llt(S);
 *         y <- solve(A, x);
 *
 *     computes the matrix-vector product $y = S^{^-1}x$, not $y = L^{-1}x$,
 *     however the Cholesky decomposition will be used to solve this more
 *     efficiently than a general matrix solve. The point of an `LLT` object
 *     is to maintain the original matrix in a decomposed form for more
 *     efficient computation. 
 */
final class LLT {    
  /*
   * Eigen internals. Eigen::LLT does not support in place matrix
   * decompositions using Eigen::Map (as of version 3.3.7, which is how Birch
   * wraps its own array buffers for use by Eigen. instead we use an
   * Eigen::Matrix type and copy the matrix to decompose into it later.
   */
  hpp{{
  Eigen::LLT<libbirch::EigenMatrix<Real>> llt;
  }}

  /**
   * Value conversion.
   */
  operator -> Real[_,_] {
    cpp{{
    return llt.reconstructedMatrix();
    }}
  }
  
  /**
   * Value assignment.
   */
  operator <- S:Real[_,_] {
    compute(S);
  }

  /**
   * Rank one update (or downdate) of the decomposition. This efficiently
   * updates the decomposition from representing $S$ to representing
   * $S + axx^\top$.
   */
  function rankUpdate(x:Real[_], a:Real) {
    cpp{{
    llt.rankUpdate(x.toEigen(), a);
    }}
  }
  
  /**
   * Decompose the matrix positive definite matrix `S` into this.
   */
  function compute(S:Real[_,_]) {
    cpp{{
    llt.compute(S.toEigen());
    }}
  }
}

/**
 * Cholesky decomposition of the symmetric positive definite matrix $S$.
 *
 * - S: The symmetric positive definite matrix $S$.
 *
 * Returns: an object representing the symmetric positive definite matrix $S$
 * in its decomposed form.
 *
 * This differs from `chol` in that `chol` returns the lower-triangular
 * Cholesky factor, while this returns the original matrix, but decomposed.
 */
function llt(S:Real[_,_]) -> LLT {
  o:LLT;
  o <- S;
  return o;
}
