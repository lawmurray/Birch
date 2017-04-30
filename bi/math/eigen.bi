/**
 * Eigen integration
 * -----------------
 */

import math.scalar;
import math.vector;
import math.matrix;
import assert;

hpp{{
#include <eigen3/Eigen/Core>

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
      EigenVectorStride(x.stride(0), 1));
}

/**
 * Convert Birch matrix to Eigen matrix.
 */
template<class Type, class Frame>
auto toEigenMatrix(const Array<Type,Frame>& X) {
  assert(X.count() == 2);
  return EigenMatrixMap<Type>(X.buf(), X.length(0), X.length(1),
      EigenMatrixStride(X.stride(1), X.lead(0)));
}

}
}}

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
