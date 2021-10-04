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
 * Addition.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> operator+(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  add(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Scalar division.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b Scalar.
 * 
 * @return %Array.
 */
template<class T, int D, class U>
Array<T,D> operator/(const Array<T,D>& A, const Scalar<U>& b) {
  Array<T,D> C(A.shape().compact());
  div(A.width(), A.height(), A.data(), A.stride(), b.data(), C.data(),
      C.stride());
  return C;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param a Scalar.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
Array<U,D> operator*(const Scalar<T>& a, const Array<U,D>& B) {
  Array<U,D> C(B.shape().compact());
  mul(C.width(), C.height(), a.data(), B.data(), B.stride(), C.data(),
      C.stride());
  return C;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b %Array.
 * 
 * @return %Array.
 */
template<class T, int D, class U>
Array<T,D> operator*(const Array<T,D>& A, const Scalar<U>& b) {
  return b*A;
}

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param x Vector.
 * 
 * @return Vector.
 */
template<class T>
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x) {
  assert(A.columns() == x.length());

  Array<T,1> y(make_shape(A.rows()));
  mul(A.rows(), A.columns(), A.data(), A.stride(), x.data(), x.stride(),
      y.data(), y.stride());
  return y;
}

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& B) {
  assert(A.columns() == B.rows());

  Array<T,2> C(make_shape(A.rows(), B.columns()));
  mul(C.rows(), C.columns(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Subtraction.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> operator-(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  sub(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

}
