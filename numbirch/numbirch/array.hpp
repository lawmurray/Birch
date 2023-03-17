/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"
#include "numbirch/array/Future.hpp"
#include "numbirch/array/Sliced.hpp"
#include "numbirch/array/Diced.hpp"
#include "numbirch/reduce.hpp"

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
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
constexpr int columns(const T& x) {
  return 1;
}

/**
 * Width of array.
 * 
 * @ingroup array
 * 
 * @see Array::width()
 */
template<class T, int D>
int width(const Array<T,D>& x) {
  return x.width();
}

/**
 * Width of scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
constexpr int width(const T& x) {
  return 1;
}

/**
 * Broadcast width of several arrays and/or scalars.
 * 
 * @ingroup array
 * 
 * @return If all arguments are scalars then 1. If one or more arguments are
 * arrays, their number of dimensions and widths must match and the width is
 * returned; scalars are broadcast in this case and do not affect the width.
 */
template<class Arg, class... Args>
int width(const Arg& arg, const Args&... args) {
  if constexpr (is_scalar_v<Arg>) {
    return width(args...);
  } else {
    assert((width(arg) == width(args...) || is_scalar_v<Args...>) &&
        "incompatible widths");
    return width(arg);
  }
}

/**
 * Height of array.
 * 
 * @ingroup array
 * 
 * @see Array::height()
 */
template<class T, int D>
int height(const Array<T,D>& x) {
  return x.height();
}

/**
 * Height of scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<class T, class = std::enable_if_t<is_scalar<T>::value,int>>
constexpr int height(const T& x) {
  return 1;
}

/**
 * Broadcast height of several arrays and/or scalars.
 * 
 * @ingroup array
 * 
 * @return If all arguments are scalars then 1. If one or more arguments are
 * arrays, their number of dimensions and heights must match and the height is
 * returned; scalars are broadcast in this case and do not affect the height.
 */
template<class Arg, class... Args>
int height(const Arg& arg, const Args&... args) {
  if constexpr (is_scalar_v<Arg>) {
    return height(args...);
  } else {
    assert((height(arg) == height(args...) || is_scalar_v<Args...>) &&
        "incompatible heights");
    return height(arg);
  }
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
 * Buffer of an array for a slice operation.
 * 
 * @ingroup array
 */
inline bool* sliced(Sliced<bool>&& x) {
  return x;
}

/**
 * Buffer of an array for a slice operation.
 * 
 * @ingroup array
 */
inline int* sliced(Sliced<int>&& x) {
  return x;
}

/**
 * Buffer of an array for a slice operation.
 * 
 * @ingroup array
 */
inline real* sliced(Sliced<real>&& x) {
  return x;
}

/**
 * Buffer of a scalar for a slice operation---just the scalar itself.
 * 
 * @ingroup array
 */
template<class T, std::enable_if_t<is_arithmetic_v<T>,int> = 0>
T sliced(const T x) {
  return x;
}

/**
 * Buffer of an array for a dice operation.
 * 
 * @ingroup array
 */
inline bool* diced(Diced<bool>&& x) {
  return x;
}

/**
 * Buffer of an array for a dice operation.
 * 
 * @ingroup array
 */
inline int* diced(Diced<int>&& x) {
  return x;
}

/**
 * Buffer of an array for a dice operation.
 * 
 * @ingroup array
 */
inline real* diced(Diced<real>&& x) {
  return x;
}

/**
 * Buffer of a scalar for a dice operation---just the scalar itself.
 * 
 * @ingroup array
 */
template<class T, std::enable_if_t<is_arithmetic_v<T>,int> = 0>
T diced(const T x) {
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
 * Construct a vector filled with a given value.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type.
 * 
 * @param x Value.
 * @param n Length.
 * 
 * @return Vector.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<value_t<T>,1> fill(const T& x, const int n);

/**
 * Gradient of fill().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Value.
 * @param n Length.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<real,0> fill_grad(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x, const int n) {
  return sum(g);
}

/**
 * Construct a matrix filled with a given value.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type.
 * 
 * @param x Value.
 * @param n Number of rows.
 * @param m Number of columns.
 * 
 * @return Matrix.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<value_t<T>,2> fill(const T& x, const int m, const int n);

/**
 * Gradient of fill().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param C Result.
 * @param x Value.
 * @param n Number of rows.
 * @param m Number of columns.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<real,0> fill_grad(const Array<real,2>& g, const Array<value_t<T>,2>& C,
    const T& x, const int m, const int n) {
  return sum(g);
}

/**
 * Construct a vector filled with a sequence of values increasing by one each
 * time.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type.
 * 
 * @param x Starting value.
 * @param n Length.
 * 
 * @return Vector.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<value_t<T>,1> iota(const T& x, const int n);

/**
 * Gradient of iota().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Starting value.
 * @param n Length.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<real,0> iota_grad(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x, const int n) {
  return sum(g);
}

/**
 * Construct diagonal matrix, filling the diagonal with a given scalar.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 * 
 * @return Diagonal matrix.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<value_t<T>,2> diagonal(const T& x, const int n);

/**
 * Gradient of diagonal().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<real,0> diagonal_grad(const Array<real,2>& g,
    const Array<value_t<T>,2>& y, const T& x, const int n) {
  return sum(g.diagonal());
}

/**
 * Construct diagonal matrix, setting the diagonal to a given vector.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Vector to assign to diagonal.
 * 
 * @return Diagonal matrix.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,2> diagonal(const Array<T,1>& x) {
  Array<T,2> y(make_shape(length(x), length(x)), T(0));
  y.diagonal() = x;
  return y;
}

/**
 * Gradient of diagonal().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Vector to assign to diagonal.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,1> diagonal_grad(const Array<real,2>& g, const Array<T,2>& y,
    const Array<T,1>& x) {
  return g.diagonal();
}

/**
 * Element of a vector.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * 
 * @param x Vector.
 * @param i Index.
 * 
 * @return Element.
 */
template<class T, class U, class = std::enable_if_t<
    is_arithmetic_v<T> && is_scalar_v<U> && is_int_v<value_t<U>>,int>>
Array<T,0> element(const Array<T,1>& x, const U& i);

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Vector.
 * @param i Index.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_scalar_v<U> && is_int_v<value_t<U>>,int>>
Array<real,1> element_grad1(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,1>& x, const U& i) {
  return single(g, i, length(x));
}

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Vector.
 * @param i Index.
 * 
 * @return Gradient with respect to @p i.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_scalar_v<U> && is_int_v<value_t<U>>,int>>
real element_grad2(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,1>& x, const U& i) {
  return real(0);
}

/**
 * Element of a matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Element.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_arithmetic_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
Array<T,0> element(const Array<T,2>& A, const U& i, const V& j);

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_arithmetic_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
Array<real,2> element_grad1(const Array<real,0>& g,
    const Array<T,0>& y, const Array<T,2>& A, const U& i, const V& j) {
  return single(g, i, j, rows(A), columns(A));
}

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Gradient with respect to @p i.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_arithmetic_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
real element_grad2(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,2>& A, const U& i, const V& j) {
  return real(0);
}

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Gradient with respect to @p j.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_arithmetic_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
real element_grad3(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,2>& A, const U& i, const V& j) {
  return real(0);
}

/**
 * Construct single-entry vector. A given element of the vector has a given
 * value, all others are zero.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * 
 * @param x Value of single entry.
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Single-entry vector.
 */
template<class T, class U, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_int_v<value_t<U>>,int>>
Array<value_t<T>,1> single(const T& x, const U& i, const int n);

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Value of single entry.
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class U, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_int_v<value_t<U>>,int>>
Array<real,0> single_grad1(const Array<real,1>& g,
    const Array<value_t<T>,1>& y, const T& x, const U& i, const int n) {
  return element(g, i);
}

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Value of single entry.
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Gradient with respect to @p i.
 */
template<class T, class U, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_int_v<value_t<U>>,int>>
real single_grad2(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x, const U& i, const int n) {
  return real(0);
}

/**
 * Construct single-entry matrix. A given element of the matrix has a given
 * value, all others are zero.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Single-entry matrix.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
    const int n);

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param A Result.
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
Array<real,0> single_grad1(const Array<real,2>& g,
    const Array<value_t<T>,2>& A, const T& x, const U& i, const V& j,
    const int m, const int n) {
  return element(g, i, j);
}

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param A Result.
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Gradient with respect to @p i.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
real single_grad2(const Array<real,2>& g, const Array<value_t<T>,2>& A,
    const T& x, const U& i, const V& j, const int m, const int n) {
  return real(0);
}

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * @tparam V Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param A Result.
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Gradient with respect to @p j.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_scalar_v<T> && is_scalar_v<U> && is_scalar_v<V> &&
    is_int_v<value_t<U>> && is_int_v<value_t<V>>,int>>
real single_grad3(const Array<real,2>& g, const Array<value_t<T>,2>& A,
    const T& x, const U& i, const V& j, const int m, const int n) {
  return real(0);
}

/**
 * Pack two arrays next to each other, concatenating their rows.
 * 
 * @ingroup array
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @p x and @p y must have the same number of rows. The result has this
 * number of rows, and a number of columns equal to the number of columns
 * of @p x plus the number of columns of @p y. The result always has two
 * dimensions.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
pack_t<T,U> pack(const T& x, const U& y) {
  assert(rows(x) == rows(y));
  [[maybe_unused]] auto r = rows(x);
  [[maybe_unused]] auto cx = columns(x);
  [[maybe_unused]] auto cy = columns(y);
  pack_t<T,U> z(make_shape(r, cx + cy));

  if constexpr (is_scalar_v<T>) {
    z.slice(1, 1) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(1, 2) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(1, std::make_pair(2, 2)) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(1, 1), std::make_pair(2, 1 + cy)) = y;
    }
  } else if constexpr (is_vector_v<T>) {
    z.slice(std::make_pair(1, r), 1) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(1, 2) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(std::make_pair(1, r), 2) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(1, r), std::make_pair(2, 1 + cy)) = y;
    }
  } else {
    static_assert(is_matrix_v<T>);
    z.slice(std::make_pair(1, r), std::make_pair(1, cx)) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(1, cx + 1) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(std::make_pair(1, r), cx + 1) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(1, r), std::make_pair(cx + 1, cx + cy)) = y;
    }
  }
  return z;
}

/**
 * Gradient of pack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
auto pack_grad1(const real_t<pack_t<T,U>>& g, const pack_t<T,U>& z, const T& x,
    const U& y) {
  assert(rows(x) == rows(y));
  [[maybe_unused]] auto r = rows(x);
  [[maybe_unused]] auto cx = columns(x);

  if constexpr (is_scalar_v<T>) {
    return g.slice(1, 1);
  } else if constexpr (is_vector_v<T>) {
    return g.slice(std::make_pair(1, r), 1);
  } else {
    static_assert(is_matrix_v<T>);
    return g.slice(std::make_pair(1, r), std::make_pair(1, cx));
  }
}

/**
 * Gradient of pack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p y.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
auto pack_grad2(const real_t<pack_t<T,U>>& g, const pack_t<T,U>& z, const T& x,
    const U& y) {
  assert(rows(x) == rows(y));
  [[maybe_unused]] auto r = rows(x);
  [[maybe_unused]] auto cx = columns(x);
  [[maybe_unused]] auto cy = columns(y);

  if constexpr (is_scalar_v<U>) {
    return g.slice(1, cx + 1);
  } else if constexpr (is_vector_v<U>) {
    return g.slice(std::make_pair(1, r), cx + 1);
  } else {
    static_assert(is_matrix_v<U>);
    return g.slice(std::make_pair(1, r), std::make_pair(cx + 1, cx + cy));
  }
}

/**
 * Stack two arrays atop one another, concatenating their columns.
 * 
 * @ingroup array
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @p x and @p y must have the same number of columns. The result has this
 * number of columns, and a number of rows equal to the number of rows of @p x
 * plus the number of rows of @p y. The result has two dimensions if at least
 * one of the arguments has two dimensions, and one dimension otherwise.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
stack_t<T,U> stack(const T& x, const U& y) {
  assert(columns(x) == columns(y));
  [[maybe_unused]] auto rx = rows(x);
  [[maybe_unused]] auto ry = rows(y);
  [[maybe_unused]] auto c = columns(x);

  if constexpr (is_scalar_v<T>) {
    if constexpr (is_scalar_v<U>) {
      stack_t<T,U> z(make_shape(2));
      z.slice(1) = x;
      z.slice(2) = y;
      return z;
    } else if constexpr (is_vector_v<U>) {
      stack_t<T,U> z(make_shape(1 + ry));
      z.slice(1) = x;
      z.slice(std::make_pair(2, 1 + ry)) = y;
      return z;
    } else {
      static_assert(is_matrix_v<U>);
      stack_t<T,U> z(make_shape(1 + ry));
      z.slice(1, 1) = x;
      z.slice(std::make_pair(2, 1 + ry), std::make_pair(1, 1)) = y;
      return z;
    }
  } else if constexpr (is_vector_v<T>) {
    if constexpr (is_scalar_v<U>) {
      stack_t<T,U> z(make_shape(rx + 1));
      z.slice(std::make_pair(1, rx)) = x;
      z.slice(rx + 1) = y;
      return z;
    } else if constexpr (is_vector_v<U>) {
      stack_t<T,U> z(make_shape(rx + ry));
      z.slice(std::make_pair(1, rx)) = x;
      z.slice(std::make_pair(rx + 1, rx + ry)) = y;
      return z;
    } else {
      static_assert(is_matrix_v<U>);
      stack_t<T,U> z(make_shape(rx + ry, 1));
      z.slice(std::make_pair(1, rx), 1) = y;
      z.slice(std::make_pair(rx + 1, rx + ry), std::make_pair(1, 1)) = y;
      return z;
    }
  } else {
    static_assert(is_matrix_v<T>);
    stack_t<T,U> z(make_shape(rx + ry, c));
    z.slice(std::make_pair(1, rx), std::make_pair(1, c)) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(rx + 1, 1) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(std::make_pair(rx + 1, rx + ry), 1) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(rx + 1, rx + ry), std::make_pair(1, c)) = y;
    }
    return z;
  }
}

/**
 * Gradient of stack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
auto stack_grad1(const real_t<stack_t<T,U>>& g, const stack_t<T,U>& z,
    const T& x, const U& y) {
  assert(columns(x) == columns(y));
  [[maybe_unused]] auto rx = rows(x);
  [[maybe_unused]] auto c = columns(x);

  if constexpr (is_scalar_v<T>) {
    if constexpr (is_matrix_v<U>) {
      return g.slice(1, 1);
    } else {
      return g.slice(1);
    }
  } else if constexpr (is_vector_v<T>) {
    if constexpr (is_matrix_v<U>) {
      return g.slice(std::make_pair(1, rx), 1);
    } else {
      return g.slice(std::make_pair(1, rx));
    }
  } else {
    static_assert(is_matrix_v<T>);
    return g.slice(std::make_pair(1, rx), std::make_pair(1, c));
  }
}

/**
 * Gradient of stack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p y.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
auto stack_grad2(const real_t<stack_t<T,U>>& g, const stack_t<T,U>& z,
    const T& x, const U& y) {
  assert(columns(x) == columns(y));
  [[maybe_unused]] auto rx = rows(x);
  [[maybe_unused]] auto ry = rows(y);
  [[maybe_unused]] auto c = columns(x);

  if constexpr (is_scalar_v<U>) {
    if constexpr (is_matrix_v<T>) {
      return g.slice(rx + 1, 1);
    } else {
      return g.slice(rx + 1);
    }
  } else if constexpr (is_vector_v<U>) {
    if constexpr (is_matrix_v<U>) {
      return g.slice(std::make_pair(rx + 1, rx + ry), 1);
    } else {
      return g.slice(std::make_pair(rx + 1, rx + ry));
    }
  } else {
    static_assert(is_matrix_v<U>);
    return g.slice(std::make_pair(rx + 1, rx + ry), std::make_pair(1, c));
  }
}

/**
 * Vectorize.
 * 
 * @ingroup array
 * 
 * @tparam T Numeric type.
 *
 * @param x Argument.
 * 
 * @return If @p x is a scalar then returns a vector with a single element. If
 * @p x is a vector then returns it as-is. If @p x is a matrix then forms a
 * vector by stacking its columns atop one another.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<value_t<T>,1> vec(const T& x);

/**
 * Gradient of vec().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
auto vec_grad(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x) {
  if constexpr (is_scalar_v<T>) {
    return g.slice(1);
  } else if constexpr (is_vector_v<T>) {
    return g;
  } else {
    return mat(g, columns(x));
  }
}

/**
 * Matrixize.
 * 
 * @ingroup array
 * 
 * @tparam T Numeric type.
 *
 * @param x Argument.
 * @param n Number of columns into which to unstack. Must be a factor of the
 * size of `x`.
 * 
 * @return If @p x is a scalar then returns a matrix with a single element. If
 * @p x is a vector then returns a matrix formed by splitting it into @p n
 * equal contiguous subvectors and unstacking them to form the columns of a
 * matrix. If @p x is a matrix then reshapes it to the given number of columns
 * as if calling `mat(vec(x), n)`.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<value_t<T>,2> mat(const T& x, const int n);

/**
 * Gradient of mat().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * @param n Number of columns into which to unstack.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
auto mat_grad(const Array<real,2>& g, const Array<value_t<T>,2>& y,
    const T& x, const int n) {
  if constexpr (is_scalar_v<T>) {
    return g.slice(1, 1);
  } else if constexpr (is_vector_v<T>) {
    return vec(g);
  } else {
    return mat(g, columns(x));
  }
}

/**
 * Vector gather.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Source.
 * @param y Indices.
 * 
 * @return Result `z`, where `z[i] = x[y[i]]`.
 * 
 * @see scatter
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,1> gather(const Array<T,1>& x, const Array<int,1>& y);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Source.
 * @param y Indices.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,1> gather_grad1(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y) {
  return scatter(g, y, length(x));
}

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Source.
 * @param y Indices.
 * 
 * @return Gradient with respect to @p y.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,1> gather_grad2(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y) {
  return fill(real(0.0), length(y));
}

/**
 * Matrix gather.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Result `C`, where `C[i,j] = A[I[i,j], J[i,j]]`.
 * 
 * @see scatter, gather
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,2> gather(const Array<T,2>& A, const Array<int,2>& I,
    const Array<int,2>& J);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param C Result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,2> gather_grad1(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J) {
  return scatter(G, C, A, I, J, rows(A), columns(A));
}

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param C Result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Gradient with respect to @p I.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,2> gather_grad2(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J) {
  return fill(real(0.0), rows(I), columns(I));
}

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param C Result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Gradient with respect to @p J.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,2> gather_grad3(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J) {
  return fill(real(0.0), rows(J), columns(J));
}

/**
 * Vector scatter.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Source.
 * @param y Indices.
 * @param n Length of result.
 * 
 * @return Result `z`, where `z[y[i]] = x[i]`.
 * 
 * In the case of collisions, e.g. `y[i] == y[j]` for some `i != j`, the
 * values are summed into the result such that `z[y[i]] == x[i] + x[j]` (c.f.
 * other libraries where the result one value or the other,
 * non-deterministically). This extends to collisions of more than two
 * elements. In the case of absence of `i` in `y`, the result is `z[i] == 0`.
 * 
 * If @p T is `bool` then the sum on collision is replaced with logical `or`
 * and the zero on absence is replaced with `false`.
 * 
 * This behavior is defined in order that the gradient of `gather` with
 * respect to its first argument is `scatter`, and conversely the gradient of
 * `scatter` with respect to its first argument is `gather`.
 * 
 * @see gather
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,1> scatter(const Array<T,1>& x, const Array<int,1>& y, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Source.
 * @param y Indices.
 * @param n Length of result.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,1> scatter_grad1(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y, const int n) {
  return gather(g, y);
}

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Source.
 * @param y Indices.
 * @param n Length of result.
 * 
 * @return Gradient with respect to @p y.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,1> scatter_grad2(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y, const int n) {
  return Array<real,1>(0, shape(y));
}

/**
 * Matrix scatter.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Result `C`, where `C[I[i,j], J[i,j]] = A[i,j]`. In the case of
 * collisions, values are summed into the result element. In the case of
 * absence, the result element is zero.
 * 
 * @see gather, scatter
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,2> scatter(const Array<T,2>& A, const Array<int,2>& I,
    const Array<int,2>& J, const int m, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param C Result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,2> scatter_grad1(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J,
    const int m, const int n) {
  return gather(G, C, A, I, J);
}

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param C Result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Gradient with respect to @p I.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,2> scatter_grad2(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J,
    const int m, const int n) {
  return fill(real(0.0), rows(I), columns(I));
}

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param C Result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Gradient with respect to @p J.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<real,2> scatter_grad3(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J,
    const int m, const int n) {
  return fill(real(0.0), rows(J), columns(J));
}

}
