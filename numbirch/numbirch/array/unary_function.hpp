/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"
#include "numbirch/memory.hpp"
#include "numbirch/numeric.hpp"

namespace numbirch {
/**
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> cholinv(const Array<T,2>& S) {
  Array<T,2> B(make_shape(S.rows(), S.columns()));
  cholinv(S.rows(), S.data(), S.stride(), B.data(), B.stride());
  return B;
}

/**
 * Inverse of a square matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> inv(const Array<T,2>& A) {
  Array<T,2> B(make_shape(A.rows(), A.columns()));
  inv(A.rows(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm of the determinant of a symmetric positive definite matrix, via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Logarithm of the determinant of `S`.
 */
template<class T>
Scalar<T> lcholdet(const Array<T,2>& S) {
  Scalar<T> b;
  lcholdet(S.rows(), S.data(), S.stride(), b.data());
  return b;
}

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * 
 * @return Logarithm of the absolute value of the determinant of `A`.
 */
template<class T>
Scalar<T> ldet(const Array<T,2>& A) {
  Scalar<T> b;
  ldet(A.rows(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> rectify(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  rectify(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Sum of elements.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return Sum of elements of the array.
 */
template<class T, int D>
Scalar<T> sum(const Array<T,D>& A) {
  Scalar<T> b;
  sum(A.width(), A.height(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Matrix trace.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * 
 * @return Trace of the matrix.
 */
template<class T>
Scalar<T> trace(const Array<T,2>& A) {
  Scalar<T> b;
  trace(A.rows(), A.columns(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Scalar product and transpose. Computes @f$B = xA^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param x Scalar.
 * @param A Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> transpose(const Array<T,2>& A) {
  Array<T,2> B(make_shape(A.columns(), A.rows()));
  transpose(B.rows(), B.columns(), A.data(), A.stride(), B.data(),
      B.stride());
  return B;
}

}
