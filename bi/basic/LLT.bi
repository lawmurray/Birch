/**
 * Cholesky decomposition of a symmetric positive definite matrix, $S = LL^T$.
 */
type LLT;

operator (X:LLT*y:Real[_]) -> Real[_] {
  return cholesky(X)*(transpose(cholesky(X))*y);
}

operator (X:LLT*Y:Real[_,_]) -> Real[_,_] {
  return matrix(X)*Y;
}

operator (X:Real[_,_]*Y:LLT) -> Real[_,_] {
  return X*matrix(Y);
}

operator (X:LLT*Y:LLT) -> Real[_,_] {
  return matrix(X)*matrix(Y);
}

operator (X:LLT + Y:Real[_,_]) -> Real[_,_] {
  return matrix(X) + Y;
}

operator (X:Real[_,_] + Y:LLT) -> Real[_,_] {
  return X + matrix(Y);
}

operator (X:LLT + Y:LLT) -> LLT {
  return rank_update(X, matrix(Y));
}

operator (X:LLT - Y:Real[_,_]) -> Real[_,_] {
  return matrix(X) - Y;
}

operator (X:Real[_,_] - Y:LLT) -> Real[_,_] {
  return X - matrix(Y);
}

operator (X:LLT - Y:LLT) -> Real[_,_] {
  return matrix(X) - matrix(Y);
}

operator (x:Real*Y:LLT) -> Real[_,_] {
  return x*matrix(Y);
}

operator (X:LLT*y:Real) -> Real[_,_] {
  return matrix(X)*y;
}

operator (X:LLT/y:Real) -> Real[_,_] {
  return matrix(X)/y;
}

operator (X:LLT == Y:LLT) -> Boolean {
  return matrix(X) == matrix(Y);
}

operator (X:LLT != Y:LLT) -> Boolean {
  return matrix(X) != matrix(Y);
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
 * Cholesky decomposition of the symmetric positive definite matrix $S$
 * (identity function).
 */
function llt(S:LLT) -> LLT {
  return S;
}

/**
 * Rank one update of a Cholesky decomposition.
 *
 * - S: Existing Cholesky decomposition of the symmetric positive definite
 *      matrix $S$.
 * - x: Vector $x$.
 *
 * Returns: A new Cholesky decomposition of the symmetric positive definite
 * matrix $S + xx^\top$.
 */
function rank_update(S:LLT, x:Real[_]) -> LLT {
  assert rows(S) == length(x);
  cpp{{
  auto A = S;
  A.rankUpdate(x.toEigen(), 1.0);
  return A;
  }}
}

/**
 * Rank $k$ update of a Cholesky decomposition.
 *
 * - S: Existing Cholesky decomposition of the symmetric positive definite
 *      matrix $S$.
 * - X: Matrix $X$.
 *
 * Returns: A new Cholesky decomposition of the symmetric positive definite
 * matrix $S + XX^\top$.
 *
 * The computation is performed as $k$ separate rank-1 updates using the
 * columns of `X
 */
function rank_update(S:LLT, X:Real[_,_]) -> LLT {
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
    A.rankUpdate(x.toEigen(), 1.0);
    }}
  }
  return A;
}

/**
 * Rank one downdate of a Cholesky decomposition.
 *
 * - S: Existing Cholesky decomposition of the symmetric positive definite
 *      matrix $S$.
 * - x: Vector $x$.
 *
 * Returns: A new Cholesky decomposition of the symmetric positive definite
 * matrix $S - xx^\top$.
 */
function rank_downdate(S:LLT, x:Real[_]) -> LLT {
  assert rows(S) == length(x);
  cpp{{
  auto A = S;
  A.rankUpdate(x.toEigen(), -1.0);
  return A;
  }}
}

/**
 * Rank $k$ downdate of a Cholesky decomposition.
 *
 * - S: Existing Cholesky decomposition of the symmetric positive definite
 *      matrix $S$.
 * - X: Matrix $X$.
 *
 * Returns: A new Cholesky decomposition of the symmetric positive definite
 * matrix $S - XX^\top$.
 *
 * The computation is performed as $k$ separate rank-1 downdates using the
 * columns of `X
 */
function rank_downdate(S:LLT, X:Real[_,_]) -> LLT {
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
    A.rankUpdate(x.toEigen(), -1.0);
    }}
  }
  return A;
}
