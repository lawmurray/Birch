/**
 * Eigen integration
 * -----------------
 */

import math.scalar;
import math.vector;
import math.matrix;
import assert;

cpp{{
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>

namespace bi {

using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
template<class Type>
using EigenVector = Eigen::Matrix<Type,Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;
template<class Type>
using EigenVectorMap = Eigen::Map<EigenVector<Type>,Eigen::Aligned128,EigenVectorStride>;

using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
template<class Type>
using EigenMatrix = Eigen::Matrix<Type,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor,Eigen::Dynamic,Eigen::Dynamic>;
template<class Type>
using EigenMatrixMap = Eigen::Map<EigenMatrix<Type>,Eigen::Aligned128,EigenMatrixStride>;

/**
 * Convert Birch vector to Eigen vector.
 */
template<class Type, class Frame>
auto toEigenVector(const Array<Type,Frame>& x) {
  assert(x.count() == 1);
  return EigenVectorMap<Type>(x.buf(), x.length(0), 1,
      EigenVectorStride(1, x.stride(0)));
}

/**
 * Convert Birch matrix to Eigen matrix.
 */
template<class Type, class Frame>
auto toEigenMatrix(const Array<Type,Frame>& X) {
  assert(X.count() == 2);
  return EigenMatrixMap<Type>(X.buf(), X.length(0), X.length(1),
      EigenMatrixStride(X.lead(1), X.stride(1)));
}

}
}}

/**
 * Negation
 * --------
 */
function -x:Real[_] -> Real[_] {
  z:Real[length(x)];
  cpp{{
  toEigenVector(z) = -toEigenVector(x);
  }}
  return z;
}

function -X:Real[_,_] -> Real[_,_] {
  Z:Real[rows(X),columns(X)];
  cpp{{
  toEigenMatrix(Z) = -toEigenMatrix(X);
  }}
  return Z;
}

/**
 * Addition
 * --------
 */
function x:Real[_] + y:Real[_] -> Real[_] {
  assert(length(x) == length(y));
  
  z:Real[length(x)];
  cpp{{
  toEigenVector(z) = toEigenVector(x) + toEigenVector(y);
  }}
  return z;
}

function X:Real[_,_] + Y:Real[_,_] -> Real[_,_] {
  assert(rows(X) == rows(Y) && columns(X) == columns(Y));
  
  Z:Real[rows(X), columns(X)];
  cpp{{
  toEigenMatrix(Z) = toEigenMatrix(X) + toEigenMatrix(Y);
  }}
  return Z;
}

/**
 * Subtraction
 * -----------
 */
function x:Real[_] - y:Real[_] -> Real[_] {
  assert(length(x) == length(y));
  
  z:Real[length(x)];
  cpp{{
  toEigenVector(z) = toEigenVector(x) - toEigenVector(y);
  }}
  return z;
}

function X:Real[_,_] - Y:Real[_,_] -> Real[_,_] {
  assert(rows(X) == rows(Y) && columns(X) == columns(Y));
  
  Z:Real[rows(X), columns(X)];
  cpp{{
  toEigenMatrix(Z) = toEigenMatrix(X) - toEigenMatrix(Y);
  }}
  return Z;
}

/**
 * Multiplication
 * --------------
 */
function x:Real*y:Real[_] -> Real[_] {
  z:Real[length(y)];
  cpp{{
  toEigenVector(z) = x * toEigenVector(y);
  }}
  return z;
}

function x:Real[_]*y:Real -> Real[_] {
  z:Real[length(x)];
  cpp{{
  toEigenVector(z) = toEigenVector(x) * y;
  }}
  return z;
}

function x:Real*Y:Real[_,_] -> Real[_,_] {
  Z:Real[rows(Y),columns(Y)];
  cpp{{
  toEigenVector(Z) = x * toEigenMatrix(Y);
  }}
  return Z;
}

function X:Real[_,_]*y:Real -> Real[_,_] {
  Z:Real[rows(X),columns(X)];
  cpp{{
  toEigenMatrix(Z) = toEigenMatrix(X) * y;
  }}
  return Z;
}

function x:Real[_]*y:Real[_] -> Real[_] {
  assert(length(y) == 1);
  
  z:Real[length(x)];
  cpp{{
  toEigenVector(z) = toEigenVector(x) * toEigenVector(y);
  }}
  return z;
}

function X:Real[_,_]*y:Real[_] -> Real[_] {
  assert(columns(X) == length(y));
  
  z:Real[rows(X)];
  cpp{{
  toEigenVector(z) = toEigenMatrix(X) * toEigenVector(y);
  }}
  return z;
}

function x:Real[_]*Y:Real[_,_] -> Real[_,_] {
  assert(1 == rows(Y));
  
  Z:Real[length(x),columns(Y)];
  cpp{{
  toEigenMatrix(Z) = toEigenVector(x) * toEigenMatrix(Y);
  }}
  return Z;
}

function X:Real[_,_]*Y:Real[_,_] -> Real[_,_] {
  assert(columns(X) == rows(Y));
  
  Z:Real[rows(X),columns(Y)];
  cpp{{
  toEigenMatrix(Z) = toEigenMatrix(X) * toEigenMatrix(Y);
  }}
  return Z;
}

/**
 * Division
 * --------
 */
function x:Real[_]/y:Real -> Real[_] {
  z:Real[length(x)];
  cpp{{
  toEigenVector(z) = toEigenVector(x) / y;
  }}
  return z;
}

function X:Real[_,_]/y:Real -> Real[_,_] {
  Z:Real[rows(X),columns(X)];
  cpp{{
  toEigenMatrix(Z) = toEigenMatrix(X) / y;
  }}
  return Z;
}

/**
 * Other vector operations
 * -----------------------
 */
/**
 * Norm of a vector.
 */
function norm(x:Real[_]) -> Real {
  cpp{{
  return toEigenVector(x).norm();
  }}
}

/**
 * Squared norm of a vector.
 */
function squaredNorm(x:Real[_]) -> Real {
  cpp{{
  return toEigenVector(x).squaredNorm();
  }}
}

/**
 * Other matrix operations
 * -----------------------
 */

/**
 * Determinant of a matrix.
 */
function determinant(X:Real[_,_]) -> Real {
  cpp{{
  return toEigenMatrix(X).determinant();
  }}
}

/**
 * Transpose of a matrix.
 */
function transpose(X:Real[_,_]) -> Real[_,_] {
  Y:Real[columns(X),rows(X)];
  
  cpp{{
  toEigenMatrix(Y) = toEigenMatrix(X).transpose();
  }}
  return Y;
}

/**
 * `LL^T` Cholesky decomposition of a matrix.
 */
function llt(X:Real[_,_]) -> Real[_,_] {
  assert(rows(X) == columns(X));
  
  L:Real[rows(X),columns(X)];
  cpp{{
  toEigenMatrix(L) = toEigenMatrix(X).llt().matrixL();
  }}
  return L;
}

/**
 * Inverse of a matrix.
 */
function inverse(X:Real[_,_]) -> Real[_,_] {
  assert(rows(X) == columns(X));
  
  invX:Real[rows(X),columns(X)];
  cpp{{
  toEigenMatrix(invX) = toEigenMatrix(X).inverse();
  }}
  return invX;
}

/**
 * Solve a system of equations.
 */
function solve(X:Real[_,_], y:Real[_]) -> Real[_] {
  assert(columns(X) == length(y));
  
  z:Real[rows(X)];
  cpp{{
  toEigenVector(z) = toEigenMatrix(X).colPivHouseholderQr().solve(toEigenVector(y));
  }}
  return z;
}

/**
 * Solve a system of equations.
 */
function solve(X:Real[_,_], Y:Real[_,_]) -> Real[_,_] {
  assert(columns(X) == rows(Y));
  
  Z:Real[rows(Y),columns(Y)];
  cpp{{
  toEigenMatrix(Z) = toEigenMatrix(X).colPivHouseholderQr().solve(toEigenMatrix(Y));
  }}
  return Z;
}
