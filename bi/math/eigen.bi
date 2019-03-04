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
operator (x:Real[_]*y:Real[_]) -> Real[_];
operator (X:Real[_,_]*y:Real[_]) -> Real[_];
operator (x:Real[_]*Y:Real[_,_]) -> Real[_,_];
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
operator (x:Integer[_]*y:Integer[_]) -> Integer[_];
operator (X:Integer[_,_]*y:Integer[_]) -> Integer[_];
operator (x:Integer[_]*Y:Integer[_,_]) -> Integer[_,_];
operator (X:Integer[_,_]*Y:Integer[_,_]) -> Integer[_,_];

/**
 * Dot product of a vector with itself.
 */
function dot(x:Real[_]) -> Real;

/**
 * Dot product of a vector with itself.
 */
function dot(x:Integer[_]) -> Integer;

/**
 * Dot product of one vector with another.
 */
function dot(x:Real[_], y:Real[_]) -> Real;

/**
 * Dot product of one vector with another.
 */
function dot(x:Integer[_], y:Integer[_]) -> Integer;

/**
 * Dot product of a vector with each of the columns of a matrix, to produce
 * matrix with a single row.
 */
function dot(x:Real[_], Y:Real[_,_]) -> Real[_,_] {
  Z:Real[1,columns(Y)];
  for j:Integer in 1..columns(Y) {
    Z[1,j] <- dot(x, Y[1..rows(Y),j]);
  }
  return Z;
}

/**
 * Dot product of a vector with each of the columns of a matrix, to produce
 * matrix with a single row.
 */
function dot(x:Integer[_], Y:Integer[_,_]) -> Integer[_,_] {
  Z:Integer[1,columns(Y)];
  for j:Integer in 1..columns(Y) {
    Z[1,j] <- dot(x, Y[1..rows(Y),j]);
  }
  return Z;
}

/**
 * Transpose of a matrix.
 */
function trans(X:Real[_,_]) -> Real[_,_];

/**
 * Transpose of a matrix.
 */
function trans(X:Integer[_,_]) -> Integer[_,_];

/**
 * Transpose of a matrix.
 */
function trans(X:Boolean[_,_]) -> Boolean[_,_];

/**
 * Transpose of a column vector into a row vector.
 */
function trans(x:Real[_]) -> Real[_,_];

/**
 * Transpose of a column vector into a row vector.
 */
function trans(x:Integer[_]) -> Integer[_,_];

/**
 * Transpose of a column vector into a row vector.
 */
function trans(x:Boolean[_]) -> Boolean[_,_];

/**
 * Norm of a vector.
 */
function norm(x:Real[_]) -> Real;

/**
 * Determinant of a matrix.
 */
function det(X:Real[_,_]) -> Real;

/**
 * Inverse of a matrix.
 */
function inv(X:Real[_,_]) -> Real[_,_];

/**
 * Solve a system of equations.
 */
function solve(X:Real[_,_], y:Real[_]) -> Real[_] {
  assert columns(X) == length(y);
  
  z:Real[rows(X)];
  cpp{{
  z_.toEigen().noalias() = X_.toEigen().householderQr().solve(y_.toEigen());
  }}
  return z;
}

/**
 * Solve a system of equations.
 */
function solve(X:Real[_,_], Y:Real[_,_]) -> Real[_,_] {
  assert columns(X) == rows(Y);
  
  Z:Real[rows(Y),columns(Y)];
  cpp{{
  Z_.toEigen().noalias() = X_.toEigen().householderQr().solve(Y_.toEigen());
  }}
  return Z;
}

/**
 * Cholesky decomposition of a matrix, $X = LL^{\top}$.
 *
 * Returns: the lower-triangular factor $L$.
 */
function chol(X:Real[_,_]) -> Real[_,_] {
  assert rows(X) == columns(X);
  
  L:Real[rows(X),columns(X)];
  cpp{{
  L_.toEigen() = X_.toEigen().llt().matrixL();
  }}
  return L;
}

/**
 * Determinant of a symmetric positive-definite matrix, computed using the
 * Cholesky decomposition.
 */
function choldet(X:Real[_,_]) -> Real {
  L:Real[_,_] <- chol(X);
  n:Integer <- rows(X);
  d:Real <- 1.0;
  for (i:Integer in 1..n) {
    d <- d*L[i,i];
  }
  return d*d;
}

/**
 * Logarithm of the determinant of a symmetric positive-definite matrix,
 * computed using the Cholesky decomposition.
 */
function lcholdet(X:Real[_,_]) -> Real {
  L:Real[_,_] <- chol(X);
  n:Integer <- rows(X);
  d:Real <- 0.0;
  for (i:Integer in 1..n) {
    d <- d + log(L[i,i]);
  }
  return 2.0*d;
}

/**
 * Inverse of a symmetric positive definite matrix via the Cholesky
 * decomposition.
 */
function cholinv(X:Real[_,_]) -> Real[_,_] {
  return cholsolve(X, identity(rows(X)));
}

/**
 * Solve a system of equations with a symmetric positive definite matrix via
 * the Cholesky decomposition.
 */
function cholsolve(X:Real[_,_], y:Real[_]) -> Real[_] {
  assert columns(X) == length(y);
  
  z:Real[rows(X)];
  cpp{{
  z_.toEigen().noalias() = X_.toEigen().llt().solve(y_.toEigen());
  }}
  return z;
}

/**
 * Solve a system of equations with a symmetric positive definite matrix via
 * the Cholesky decomposition.
 */
function cholsolve(X:Real[_,_], Y:Real[_,_]) -> Real[_,_] {
  assert columns(X) == rows(Y);
  
  Z:Real[rows(Y),columns(Y)];
  cpp{{
  Z_.toEigen().noalias() = X_.toEigen().llt().solve(Y_.toEigen());
  }}
  return Z;
}
