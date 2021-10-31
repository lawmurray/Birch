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
#include "numbirch/type.hpp"

namespace numbirch {
/**
 * Addition.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator+(const Array<T,D>& A,
    const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<typename promote<T,U>::type,D> C(A.shape().compact());
  add(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Addition.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U, std::enable_if_t<std::is_arithmetic<T>::value &&
    std::is_arithmetic<U>::value,int> = 0>
PURE Scalar<typename promote<T,U>::type> operator+(const Scalar<T>& x,
    const U& y) {
  return x + Scalar<U>(y);
}

/**
 * Addition.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U, std::enable_if_t<std::is_arithmetic<T>::value &&
    std::is_arithmetic<U>::value,int> = 0>
PURE Scalar<typename promote<T,U>::type> operator+(const T& x,
    const Scalar<U>& y) {
  return Scalar<T>(x) + y;
}

/**
 * Scalar division.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b Scalar.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator/(const Array<T,D>& A,
    const Scalar<U>& b) {
  Array<typename promote<T,U>::type,D> C(A.shape().compact());
  div(A.width(), A.height(), A.data(), A.stride(), b.data(), C.data(),
      C.stride());
  return C;
}

/**
 * Scalar division.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b %Scalar.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator/(const Array<T,D>& A,
    const U& b) {
  return A/Scalar<U>(b);
}

/**
 * Scalar division.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Scalar<typename promote<T,U>::type> operator/(const T& x, const Scalar<U>& y) {
  return Scalar<T>(x)/y;
}

/**
 * Equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Array<bool,D> operator==(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  equal(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator==(const Scalar<T>& x, const T& y) {
  return x == Scalar<T>(y);
}

/**
 * Equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator==(const T& x, const Scalar<T>& y) {
  return Scalar<T>(x) == y;
}

/**
 * Greater than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Array<bool,D> operator>(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  greater(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Greater than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator>(const Scalar<T>& x, const T& y) {
  return x > Scalar<T>(y);
}

/**
 * Greater than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator>(const T& x, const Scalar<T>& y) {
  return Scalar<T>(x) > y;
}

/**
 * Greater than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Array<bool,D> operator>=(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  greater_or_equal(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Greater than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator>=(const Scalar<T>& x, const T& y) {
  return x >= Scalar<T>(y);
}

/**
 * Greater than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator>=(const T& x, const Scalar<T>& y) {
  return Scalar<T>(x) >= y;
}

/**
 * Less than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Array<bool,D> operator<(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  less(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Less than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator<(const Scalar<T>& x, const T& y) {
  return x < Scalar<T>(y);
}

/**
 * Less than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator<(const T& x, const Scalar<T>& y) {
  return Scalar<T>(x) < y;
}

/**
 * Less than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Array<bool,D> operator<=(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  less_or_equal(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Less than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator<=(const Scalar<T>& x, const T& y) {
  return x <= Scalar<T>(y);
}

/**
 * Less than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator<=(const T& x, const Scalar<T>& y) {
  return Scalar<T>(x) <= y;
}

/**
 * Logical `and`.
 * 
 * @ingroup array
 * 
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<int D>
PURE Array<bool,D> operator&&(const Array<bool,D>& A, const Array<bool,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  logical_and(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Logical `and`.
 * 
 * @ingroup array
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
PURE inline Scalar<bool> operator&&(const Scalar<bool>& x, const bool& y) {
  return x && Scalar<bool>(y);
}

/**
 * Logical `and`.
 * 
 * @ingroup array
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
PURE inline Scalar<bool> operator&&(const bool& x, const Scalar<bool>& y) {
  return Scalar<bool>(x) && y;
}

/**
 * Logical `or`.
 * 
 * @ingroup array
 * 
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<int D>
PURE Array<bool,D> operator||(const Array<bool,D>& A, const Array<bool,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  logical_or(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Logical `or`.
 * 
 * @ingroup array
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
PURE inline Scalar<bool> operator||(const Scalar<bool>& x, const bool& y) {
  return x || Scalar<bool>(y);
}

/**
 * Logical `or`.
 * 
 * @ingroup array
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
PURE inline Scalar<bool> operator||(const bool& x, const Scalar<bool>& y) {
  return Scalar<bool>(x) || y;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param a %Scalar.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator*(const Scalar<T>& a,
    const Array<U,D>& B) {
  Array<typename promote<T,U>::type,D> C(B.shape().compact());
  mul(C.width(), C.height(), a.data(), B.data(), B.stride(), C.data(),
      C.stride());
  return C;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param a Scalar.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator*(const T& a,
    const Array<U,D>& B) {
  return Scalar<T>(a)*B;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b %Array.
 * 
 * @return %Array.
 * 
 * @note Disabled for `D == 0` as signature would be identical with
 * `operator*(const Scalar<T>&, const Array<T,D>&)`.
 */
template<class T, class U, int D, std::enable_if_t<(D > 0) &&
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator*(const Array<T,D>& A,
    const Scalar<U>& b) {
  return b*A;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator*(const Array<T,D>& A,
    const U& b) {
  return b*A;
}

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param A %Matrix.
 * @param x %Vector.
 * 
 * @return %Vector.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Vector<T> operator*(const Matrix<T>& A, const Vector<T>& x) {
  assert(A.columns() == x.length());

  Vector<T> y(make_shape(A.rows()));
  mul(A.rows(), A.columns(), A.data(), A.stride(), x.data(), x.stride(),
      y.data(), y.stride());
  return y;
}

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param A %Matrix.
 * @param B %Matrix.
 * 
 * @return %Matrix.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
  assert(A.columns() == B.rows());

  Matrix<T> C(make_shape(A.rows(), B.columns()));
  mul(C.rows(), C.columns(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Not equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Array<bool,D> operator!=(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  not_equal(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Not equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator!=(const Scalar<T>& x, const T& y) {
  return x != Scalar<T>(y);
}

/**
 * Not equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<bool> operator!=(const T& x, const Scalar<T>& y) {
  return Scalar<T>(x) != y;
}

/**
 * Subtraction.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Array<typename promote<T,U>::type,D> operator-(const Array<T,D>& A,
    const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<typename promote<T,U>::type,D> C(A.shape().compact());
  sub(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Subtraction.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Scalar<typename promote<T,U>::type> operator-(const Scalar<T>& x,
    const U& y) {
  return x - Scalar<T>(y);
}

/**
 * Subtraction.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,int> = 0>
PURE Scalar<typename promote<T,U>::type> operator-(const T& x,
    const Scalar<U>& y) {
  return Scalar<T>(x) - y;
}

}
