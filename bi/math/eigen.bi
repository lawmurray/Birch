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
function dot(x:Integer[_]) -> Integer;

/**
 * Transpose of a matrix.
 */
function trans(X:Real[_,_]) -> Real[_,_];
function trans(X:Integer[_,_]) -> Integer[_,_];
function trans(X:Boolean[_,_]) -> Boolean[_,_];

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

/*
 * for the below functions, need to assign the result to a new matrix, as, it
 * seems, they return a view of a matrix that will be destroyed on return
 */
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
 * Solve a system of equations.
 */
function solve(X:Real[_,_], y:Real[_]) -> Real[_] {
  assert columns(X) == length(y);
  
  z:Real[rows(X)];
  cpp{{
  z_.toEigen() = X_.toEigen().colPivHouseholderQr().solve(y_.toEigen());
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
  Z_.toEigen() = X_.toEigen().colPivHouseholderQr().solve(Y_.toEigen());
  }}
  return Z;
}
