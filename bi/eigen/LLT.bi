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
final class LLT(n:Integer) {
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
   * Decompose the matrix positive definite matrix `S` into this.
   */
  function compute(S:Real[_,_]) {
    cpp{{
    llt.compute(S.toEigen());
    }}
  }

  /**
   * Rank one update (or downdate) of a Cholesky decomposition.
   *
   * - x: Vector.
   * - a: Scalar. Positive for an update, negative for a downdate.
   *
   * Updates the symmetric positive definite matrix to $S + axx^\top$ with an
   * efficient update of its decomposition.
   */
  function update(x:Real[_], a:Real) {
    cpp{{
    llt.rankUpdate(x.toEigen(), a);
    }}
  }
}

operator (X:LLT*y:Real[_]) -> Real[_] {
  cpp{{
  return X->llt.matrixL()*(X->llt.matrixU()*y.toEigen()).eval();
  }}
}

operator (X:LLT*Y:Real[_,_]) -> Real[_,_] {
  cpp{{
  return X->llt.matrixL()*(X->llt.matrixU()*Y.toEigen()).eval();
  }}
}

operator (X:Real[_,_]*Y:LLT) -> Real[_,_] {
  cpp{{
  return X.toEigen()*Y->llt.matrixL()*Y->llt.matrixU();
  }}
}

/**
 * Number of rows of a symmetric positive definite matrix.
 */
function rows(X:LLT) -> Integer64 {
  cpp{{
  return X->llt.rows();
  }}
}

/**
 * Number of columns of a symmetric positive definite matrix.
 */
function columns(X:LLT) -> Integer64 {
  cpp{{
  return X->llt.cols();
  }}
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
  assert rows(S) == columns(S);
  A:LLT(rows(S));
  A.compute(S);
  return A;
}

/**
 * Rank one update (or downdate) of a Cholesky decomposition.
 *
 * - S: Existing Cholesky decomposition of the symmetric positive definite
 *      matrix $S$.
 * - x: Vector $x$.
 * - a: Scalar $a$. Positive for an update, negative for a downdate.
 *
 * Returns: A new Cholesky decomposition of the symmetric positive definite
 * matrix $S + axx^\top$.
 */
function rank_update(S:LLT, x:Real[_], a:Real) -> LLT {
  A:LLT(rows(S));
  cpp{{
  A->llt = S->llt;
  }}
  A.update(x, a);
  return A;
}

/**
 * Rank $k$ update (or downdate) of a Cholesky decomposition.
 *
 * - S: Existing Cholesky decomposition of the symmetric positive definite
 *      matrix $S$.
 * - X: Matrix $X$.
 * - a: Scalar $a$. Positive for an update, negative for a downdate.
 *
 * Returns: A new Cholesky decomposition of the symmetric positive definite
 * matrix $S + aXX^\top$.
 *
 * The computation is performed as $k$ separate rank-1 updates using the
 * columns of `X
 */
function rank_update(S:LLT, X:Real[_,_], a:Real) -> LLT {
  A:LLT(rows(S));
  cpp{{
  A->llt = S->llt;
  }}
  auto R <- rows(X);
  auto C <- columns(X);
  for auto j in 1..C {
    A.update(X[1..R,j], a);
  }
  return A;
}

/**
 * Trace of a symmetric positive-definite matrix.
 */
function trace(S:LLT) -> Real {
  auto L <- cholesky(S);
  auto n <- rows(S);
  auto tr <- 0.0;
  for auto i in 1..n {
    auto l <- L[i,i];
    tr <- tr + l*l;
  }
  return tr;
}

/**
 * Determinant of a symmetric positive-definite matrix.
 */
function det(S:LLT) -> Real {
  auto L <- cholesky(S);
  auto n <- rows(S);
  auto d <- 1.0;
  for auto i in 1..n {
    d <- d*L[i,i];
  }
  return d*d;
}

/**
 * Logarithm of the determinant of a symmetric positive-definite matrix.
 */
function ldet(S:LLT) -> Real {
  auto L <- cholesky(S);
  auto n <- rows(S);
  auto d <- 0.0;
  for auto i in 1..n {
    d <- d + log(L[i,i]);
  }
  return 2.0*d;
}

/**
 * Inverse of a symmetric positive definite matrix.
 */
function inv(S:LLT) -> Real[_,_] {
  cpp{{
  return S->llt.solve(libbirch::EigenMatrix<bi::type::Real>::Identity(
      S->llt.rows(), S->llt.cols()));
  }}
}

/**
 * Solve a system of equations with a symmetric positive definite matrix.
 */
function solve(S:LLT, y:Real[_]) -> Real[_] {
  cpp{{
  return S->llt.solve(y.toEigen());
  }}
}

/**
 * Solve a system of equations with a symmetric positive definite matrix.
 */
function solve(S:LLT, Y:Real[_,_]) -> Real[_,_] {
  cpp{{
  return S->llt.solve(Y.toEigen());
  }}
}

/**
 * Cholesky factor of a matrix, $X = LL^{\top}$.
 *
 * Returns: the lower-triangular factor $L$.
 */
function cholesky(S:LLT) -> Real[_,_] {
  L:Real[_,_];
  cpp{{
  L.toEigen() = S->llt.matrixL();
  }}
  return L;
}
