/*
 * Birch uses the Eigen library <https://eigen.tuxfamily.org> for linear
 * algebra support. Eigen is tightly integrated with support from libbirch,
 * in order to preserve the lazy evaluation of Eigen that is a critical
 * feature of its performance.
 */

operator (x:Real + y:Real[_]) -> Real[_];
operator (x:Real[_] + y:Real) -> Real[_];
operator (x:Real + Y:Real[_,_]) -> Real[_,_];
operator (X:Real[_,_] + y:Real) -> Real[_,_];
operator (x:Real - y:Real[_]) -> Real[_];
operator (x:Real[_] - y:Real) -> Real[_];
operator (x:Real - Y:Real[_,_]) -> Real[_,_];
operator (X:Real[_,_] - y:Real) -> Real[_,_];
operator (x:Real*y:Real[_]) -> Real[_];
operator (x:Real[_]*y:Real) -> Real[_];
operator (x:Real*Y:Real[_,_]) -> Real[_,_];
operator (X:Real[_,_]*y:Real) -> Real[_,_];
operator (x:Real[_]/y:Real) -> Real[_];
operator (X:Real[_,_]/y:Real) -> Real[_,_];
operator (x:Real[_] == y:Real[_]) -> Boolean;
operator (x:Real[_] != y:Real[_]) -> Boolean;
operator (X:Real[_,_] == Y:Real[_,_]) -> Boolean;
operator (X:Real[_,_] != Y:Real[_,_]) -> Boolean;
operator (-x:Real[_]) -> Real[_];
operator (+x:Real[_]) -> Real[_];
operator (-X:Real[_,_]) -> Real[_,_];
operator (+X:Real[_,_]) -> Real[_,_];
operator (x:Real[_] + y:Real[_]) -> Real[_];
operator (x:Real[_] - y:Real[_]) -> Real[_];
operator (X:Real[_,_] + Y:Real[_,_]) -> Real[_,_];
operator (X:Real[_,_] - Y:Real[_,_]) -> Real[_,_];
operator (X:Real[_,_]*y:Real[_]) -> Real[_];
operator (X:Real[_,_]*Y:Real[_,_]) -> Real[_,_];

operator (x:Integer + y:Integer[_]) -> Integer[_];
operator (x:Integer[_] + y:Integer) -> Integer[_];
operator (x:Integer + Y:Integer[_,_]) -> Integer[_,_];
operator (X:Integer[_,_] + y:Integer) -> Integer[_,_];
operator (x:Integer - y:Integer[_]) -> Integer[_];
operator (x:Integer[_] - y:Integer) -> Integer[_];
operator (x:Integer - Y:Integer[_,_]) -> Integer[_,_];
operator (X:Integer[_,_] - y:Integer) -> Integer[_,_];
operator (x:Integer*y:Integer[_]) -> Integer[_];
operator (x:Integer[_]*y:Integer) -> Integer[_];
operator (x:Integer*Y:Integer[_,_]) -> Integer[_,_];
operator (X:Integer[_,_]*y:Integer) -> Integer[_,_];
operator (x:Integer[_]/y:Integer) -> Integer[_];
operator (X:Integer[_,_]/y:Integer) -> Integer[_,_];
operator (x:Integer[_] == y:Integer[_]) -> Boolean;
operator (x:Integer[_] != y:Integer[_]) -> Boolean;
operator (X:Integer[_,_] == Y:Integer[_,_]) -> Boolean;
operator (X:Integer[_,_] != Y:Integer[_,_]) -> Boolean;
operator (-x:Integer[_]) -> Integer[_];
operator (+x:Integer[_]) -> Integer[_];
operator (-X:Integer[_,_]) -> Integer[_,_];
operator (+X:Integer[_,_]) -> Integer[_,_];
operator (x:Integer[_] + y:Integer[_]) -> Integer[_];
operator (x:Integer[_] - y:Integer[_]) -> Integer[_];
operator (X:Integer[_,_] + Y:Integer[_,_]) -> Integer[_,_];
operator (X:Integer[_,_] - Y:Integer[_,_]) -> Integer[_,_];
operator (X:Integer[_,_]*y:Integer[_]) -> Integer[_];
operator (X:Integer[_,_]*Y:Integer[_,_]) -> Integer[_,_];

/**
 * Dot product of vector with itself.
 */
function dot(x:Real[_]) -> Real;

/**
 * Dot product of vector with another.
 */
function dot(x:Real[_], y:Real[_]) -> Real;

/**
 * Norm of a vector.
 */
function norm(x:Real[_]) -> Real;

/**
 * Element-wise square root of a vector.
 */
function sqrt(x:Real[_]) -> Real[_];

/**
 * Transpose of a matrix.
 */
function transpose(X:Real[_,_]) -> Real[_,_];

/**
 * Transpose of a column vector into a row vector.
 */
function transpose(x:Real[_]) -> Real[_,_];

/**
 * Diagonal matrix from vector.
 */
function diagonal(x:Real[_]) -> Real[_,_];

/**
 * Diagonal of a matrix, as a vector.
 */
function diagonal(X:Real[_,_]) -> Real[_];

/**
 * Trace of a matrix.
 */
function trace(X:Real[_,_]) -> Real;

/**
 * Determinant of a matrix.
 */
function det(X:Real[_,_]) -> Real;

/**
 * Logarithm of the determinant of a matrix.
 */
function ldet(X:Real[_,_]) -> Real {
  ///@todo Use Eigen LU module
  return log(det(X));
}

/**
 * Cholesky factor of a positive symmetric matrix, $X = LL^{\top}$.
 *
 * Returns: the lower-triangular factor $L$.
 */
function cholesky(X:Real[_,_]) -> Real[_,_] {
  assert rows(X) == columns(X);
  
  L:Real[rows(X),columns(X)];
  cpp{{
  L.toEigen() = X.toEigen().llt().matrixL();
  }}
  return L;
}

/**
 * Inverse of a matrix.
 */
function inv(X:Real[_,_]) -> Real[_,_];

/**
 * Solve a system of equations.
 */
function solve(X:Real[_,_], y:Real[_]) -> Real[_];

/**
 * Solve a system of equations.
 */
function solve(X:Real[_,_], Y:Real[_,_]) -> Real[_,_];

/**
 * Kronecker vector-vector product.
 */
function kronecker(x:Real[_], y:Real[_]) -> Real[_];

/**
 * Kronecker matrix-vector product.
 */
function kronecker(X:Real[_,_], y:Real[_]) -> Real[_,_];

/**
 * Kronecker vector-matrix product.
 */
function kronecker(x:Real[_], Y:Real[_,_]) -> Real[_,_];

/**
 * Kronecker matrix-matrix product.
 */
function kronecker(X:Real[_,_], Y:Real[_,_]) -> Real[_,_];

/**
 * Hadamard (element-wise) vector product.
 */
function hadamard(x:Real[_], y:Real[_]) -> Real[_];

/**
 * Hadamard (element-wise) matrix product.
 */
function hadamard(X:Real[_,_], Y:Real[_,_]) -> Real[_,_];
