/**
 * @file
 * 
 * @defgroup array Array
 * Multidimensional arrays with copy-on-write and streamlined device
 * synchronization.
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"
#include "numbirch/array/Future.hpp"
#include "numbirch/macro.hpp"

namespace numbirch {
/**
 * Length of an array.
 * 
 * @ingroup array
 * 
 * @see Array::length()
 */
template<class T, int D>
int length(const Array<T,D>& x) {
  return x.length();
}

/**
 * Length of a scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr int length(const T& x) {
  return 1;
}

/**
 * Number of rows in array.
 * 
 * @ingroup array
 * 
 * @see Array::rows()
 */
template<class T, int D>
int rows(const Array<T,D>& x) {
  return x.rows();
}

/**
 * Number of rows in scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr int rows(const T& x) {
  return 1;
}

/**
 * Number of columns in array.
 * 
 * @ingroup array
 * 
 * @see Array::columns()
 */
template<class T, int D>
int columns(const Array<T,D>& x) {
  return x.columns();
}

/**
 * Number of columns in scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr int columns(const T& x) {
  return 1;
}

/**
 * Stride of an array.
 * 
 * @ingroup array
 * 
 * @see Array::stride()
 */
template<class T, int D>
int stride(const Array<T,D>& x) {
  return x.stride();
}

/**
 * Stride of a scalar---i.e. 0, although typically ignored by functions.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr int stride(const T& x) {
  return 0;
}

/**
 * Size of an array.
 * 
 * @ingroup array
 * 
 * @see Array::size()
 */
template<class T, int D>
int size(const Array<T,D>& x) {
  return x.size();
}

/**
 * Size of a scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr int size(const T& x) {
  return 1;
}

/**
 * Shape of an array.
 * 
 * @ingroup array
 * 
 * @see Array::shape()
 */
template<class T, int D>
ArrayShape<D> shape(const Array<T,D>& x) {
  return x.shape();
}

/**
 * Shape of a scalar.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
ArrayShape<0> shape(const T& x) {
  return make_shape();
}

/**
 * Buffer of an array.
 * 
 * @ingroup array
 * 
 * @see Array::data()
 */
template<class T, int D>
const T* data(const Array<T,D>& x) {
  return x.data();
}

/**
 * Buffer of an array.
 * 
 * @ingroup array
 */
template<class T, int D>
T* data(Array<T,D>& x) {
  return x.data();
}

/**
 * Buffer of a scalar---just the scalar itself.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr const T& data(const T& x) {
  return x;
}

/**
 * Do the shapes of two arrays conform?---Yes, if they have the same number of
 * dimensions and same length along them.
 * 
 * @ingroup array
 * 
 * @see Array::conforms()
 */
template<class T, int D, class U, int E>
bool conforms(const Array<T,D>& x, const Array<U,E>& y) {
  return x.conforms(y);
}

/**
 * Does the shape of an array conform with that of a scalar?---Yes, if it has
 * zero dimensions.
 * 
 * @ingroup array
 */
template<class T, int D, class U, class = std::enable_if_t<
    is_scalar<U>::value,int>>
constexpr bool conforms(const Array<T,D>& x, const U& y) {
  return D == 0;
}

/**
 * Does the shape of an array conform with that of a scalar?---Yes, if it has
 * zero dimensions.
 * 
 * @ingroup array
 */
template<class T, class U, int D, class = std::enable_if_t<
    is_scalar<T>::value,int>>
constexpr bool conforms(const T& x, const Array<U,D>& y) {
  return D == 0;
}

/**
 * Do the shapes of two scalars conform?---Yes.
 * 
 * @ingroup array
 */
template<class T, class U, class = std::enable_if_t<is_scalar<T>::value &&
    is_scalar<U>::value,int>>
constexpr bool conforms(const T& x, const U& y) {
  return true;
}

/**
 * @internal
 *
 * Element of a matrix, vector, or scalar. A scalar is identified by having
 * `ld == 0`.
 */
template<class T>
NUMBIRCH_HOST_DEVICE T& element(T* x, const int i = 0, const int j = 0,
    const int ld = 0) {
  int k = (ld == 0) ? 0 : i + j*ld;
  return x[k];
}

/**
 * @internal
 *
 * Element of a matrix, vector, or scalar. A scalar is identified by having
 * `ld == 0`.
 */
template<class T>
NUMBIRCH_HOST_DEVICE const T& element(const T* x, const int i = 0,
    const int j = 0, const int ld = 0) {
  int k = (ld == 0) ? 0 : i + j*ld;
  return x[k];
}

/**
 * @internal
 * 
 * Element of a scalar---just returns the scalar.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
NUMBIRCH_HOST_DEVICE T& element(T& x, const int i = 0, const int j = 0,
    const int ld = 0) {
  return x;
}

/**
 * @internal
 * 
 * Element of a scalar---just returns the scalar.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
NUMBIRCH_HOST_DEVICE const T& element(const T& x, const int i = 0,
    const int j = 0, const int ld = 0) {
  return x;
}

}
