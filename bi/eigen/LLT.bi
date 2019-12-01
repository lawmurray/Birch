/**
 * Cholesky decomposition of a symmetric positive definite matrix, $S = LL^T$.
 */
type LLT;

operator (X:LLT*y:Real[_]) -> Real[_] {
  assert columns(X) == length(y);
  cpp{{
  return X.matrixL()*(X.matrixU()*y.toEigen()).eval();
  }}
}

operator (X:LLT*Y:Real[_,_]) -> Real[_,_] {
  assert columns(X) == rows(Y);
  cpp{{
  return X.matrixL()*(X.matrixU()*Y.toEigen()).eval();
  }}
}

operator (X:Real[_,_]*Y:LLT) -> Real[_,_] {
  assert columns(X) == rows(Y);
  cpp{{
  return X.toEigen()*Y.matrixL()*Y.matrixU();
  }}
}

/**
 * Number of rows of a symmetric positive definite matrix.
 */
function rows(X:LLT) -> Integer64 {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of columns of a symmetric positive definite matrix.
 */
function columns(X:LLT) -> Integer64 {
  cpp{{
  return X.cols();
  }}
}

/**
 * Convert Cholesky decomposition to an ordinary matrix.
 */
function matrix(X:LLT) -> Real[_,_] {
  cpp{{
  return X.reconstructedMatrix();
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
 * !!! attention
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
function llt(S:Real[_,_]) -> LLT {
  A:LLT;
  cpp{{
  A.compute(S.toEigen());
  }}
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
  assert rows(S) == length(x);
  cpp{{
  auto A = S;
  A.rankUpdate(x.toEigen(), a);
  return A;
  }}
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
  assert rows(S) == rows(X);
  A:LLT;
  cpp{{
  A = S;
  }}
  auto R <- rows(X);
  auto C <- columns(X);
  for j in 1..C {
    auto x <- X[1..R,j];
    cpp{{
    A.rankUpdate(x.toEigen(), a);
    }}
  }
  return A;
}

/**
 * Trace of a symmetric positive-definite matrix.
 */
function trace(S:LLT) -> Real;

/**
 * Determinant of a symmetric positive-definite matrix.
 */
function det(S:LLT) -> Real;

/**
 * Logarithm of the determinant of a symmetric positive-definite matrix.
 */
function ldet(S:LLT) -> Real {
  auto L <- cholesky(S);
  auto n <- rows(S);
  auto d <- 0.0;
  for i in 1..n {
    d <- d + log(L[i,i]);
  }
  return 2.0*d;
}

/**
 * Inverse of a symmetric positive definite matrix.
 */
function inv(S:LLT) -> Real[_,_];

/**
 * Solve a system of equations with a symmetric positive definite matrix.
 */
function solve(S:LLT, y:Real[_]) -> Real[_];

/**
 * Solve a system of equations with a symmetric positive definite matrix.
 */
function solve(S:LLT, Y:Real[_,_]) -> Real[_,_];

/**
 * Cholesky factor of a matrix, $X = LL^{\top}$.
 *
 * Returns: the lower-triangular factor $L$.
 */
function cholesky(S:LLT) -> Real[_,_];
