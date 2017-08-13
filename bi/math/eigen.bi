/*
 * Birch uses the Eigen library <https://eigen.tuxfamily.org> for linear
 * algebra support. Eigen is tightly integrated with support from the Birch
 * compiler library, in order to preserve the lazy evaluation of Eigen that
 * is a critical feature of its performance.
 */
import basic;
import math.vector;
import math.matrix;

/*
 * Basic operators
 */
operator -x:Real[_] -> Real[_];
operator +x:Real[_] -> Real[_];
operator x:Real[_] + y:Real[_] -> Real[_];
operator x:Real[_] - y:Real[_] -> Real[_];

operator -X:Real[_,_] -> Real[_,_];
operator +X:Real[_,_] -> Real[_,_];
operator X:Real[_,_] + Y:Real[_,_] -> Real[_,_];
operator X:Real[_,_] - Y:Real[_,_] -> Real[_,_];

operator x:Real[_]*y:Real[_] -> Real[_];
operator X:Real[_,_]*y:Real[_] -> Real[_];
operator x:Real[_]*Y:Real[_,_] -> Real[_,_];
operator X:Real[_,_]*Y:Real[_,_] -> Real[_,_];

operator x:Real*y:Real[_] -> Real[_];
operator x:Real[_]*y:Real -> Real[_];
operator x:Real*Y:Real[_,_] -> Real[_,_];
operator X:Real[_,_]*y:Real -> Real[_,_];
operator x:Real[_]/y:Real -> Real[_];
operator X:Real[_,_]/y:Real -> Real[_,_];

/**
 * Norm of a vector.
 */
function norm(x:Real[_]) -> Real;

/**
 * Squared norm of a vector.
 */
function squaredNorm(x:Real[_]) -> Real;

/**
 * Determinant of a matrix.
 */
function determinant(X:Real[_,_]) -> Real;

/**
 * Transpose of a matrix.
 */
function transpose(X:Real[_,_]) -> Real[_,_];

/**
 * Inverse of a matrix.
 */
function inverse(X:Real[_,_]) -> Real[_,_];

/*
 * for the below functions, need to assign the result to a new matrix, as, it
 * seems, they return a view of a matrix that will be destroyed on return
 */
/**
 * `LL^T` Cholesky decomposition of a matrix.
 */
function llt(X:Real[_,_]) -> Real[_,_] {
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
