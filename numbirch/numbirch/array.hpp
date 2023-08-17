/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Future.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"
#include "numbirch/reduce.hpp"

namespace numbirch {
/**
 * Length of an array.
 * 
 * @ingroup array
 * 
 * @see Array::length()
 */
template<arithmetic T, int D>
int length(const Array<T,D>& x) {
  return x.length();
}

/**
 * Length of a scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<scalar T>
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
template<arithmetic T, int D>
int rows(const Array<T,D>& x) {
  return x.rows();
}

/**
 * Number of rows in scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<scalar T>
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
template<arithmetic T, int D>
int columns(const Array<T,D>& x) {
  return x.columns();
}

/**
 * Number of columns in scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<scalar T>
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
template<arithmetic T, int D>
int width(const Array<T,D>& x) {
  return x.width();
}

/**
 * Width of scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<scalar T>
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
    assert((width(arg) == width(args...) || all_scalar_v<Args...>) &&
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
template<arithmetic T, int D>
int height(const Array<T,D>& x) {
  return x.height();
}

/**
 * Height of scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<scalar T>
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
    assert((height(arg) == height(args...) || all_scalar_v<Args...>) &&
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
template<arithmetic T, int D>
int stride(const Array<T,D>& x) {
  return x.stride();
}

/**
 * Stride of a scalar---i.e. 0, although typically ignored by functions.
 * 
 * @ingroup array
 */
template<scalar T>
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
template<arithmetic T, int D>
int size(const Array<T,D>& x) {
  return x.size();
}

/**
 * Size of a scalar---i.e. 1.
 * 
 * @ingroup array
 */
template<scalar T>
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
template<arithmetic T, int D>
Shape<D> shape(const Array<T,D>& x) {
  return x.shape();
}

/**
 * Shape of a scalar.
 * 
 * @ingroup array
 */
template<scalar T>
Shape<0> shape(const T& x) {
  return make_shape();
}

/**
 * Buffer of an array.
 * 
 * @ingroup array
 */
template<arithmetic T, int D>
T* buffer(Array<T,D>& x) {
  return x.buffer();
}

/**
 * Buffer of an array.
 * 
 * @ingroup array
 */
template<arithmetic T, int D>
const T* buffer(const Array<T,D>& x) {
  return x.buffer();
}

/**
 * Buffer of a scalar for a slice operation---just the scalar itself.
 * 
 * @ingroup array
 */
template<arithmetic T>
T buffer(const T& x) {
  return x;
}

/**
 * Stream of an array.
 * 
 * @ingroup array
 */
template<arithmetic T, int D>
void*& stream(const Array<T,D>& x) {
  return x.stream();
}

/**
 * Stream of a scalar.
 * 
 * @ingroup array
 */
template<arithmetic T>
void* stream(const T& x) {
  return nullptr;
}

/**
 * Do the shapes of two arrays conform?---Yes, if they have the same number of
 * dimensions and same length along them.
 * 
 * @ingroup array
 * 
 * @see Array::conforms()
 */
template<arithmetic T, int D, arithmetic U, int E>
bool conforms(const Array<T,D>& x, const Array<U,E>& y) {
  return x.conforms(y);
}

/**
 * Does the shape of an array conform with that of a scalar?---Yes, if it has
 * zero dimensions.
 * 
 * @ingroup array
 */
template<arithmetic T, int D, scalar U>
constexpr bool conforms(const Array<T,D>& x, const U& y) {
  return D == 0;
}

/**
 * Does the shape of an array conform with that of a scalar?---Yes, if it has
 * zero dimensions.
 * 
 * @ingroup array
 */
template<scalar T, class U, int D>
constexpr bool conforms(const T& x, const Array<U,D>& y) {
  return D == 0;
}

/**
 * Do the shapes of two scalars conform?---Yes.
 * 
 * @ingroup array
 */
template<scalar T, scalar U>
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
template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,1> fill(const T& x, const int n);

/**
 * Gradient of fill().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Value.
 * @param n Length.
 * 
 * @return Gradient with respect to @p x.
 */
template<scalar T>
NUMBIRCH_KEEP Array<real,0> fill_grad(const Array<real,1>& g, const T& x, const int n);

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
template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,2> fill(const T& x, const int m, const int n);

/**
 * Gradient of fill().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Value.
 * @param n Number of rows.
 * @param m Number of columns.
 * 
 * @return Gradient with respect to @p x.
 */
template<scalar T>
NUMBIRCH_KEEP Array<real,0> fill_grad(const Array<real,2>& g, const T& x, const int m,
    const int n);

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
template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,1> iota(const T& x, const int n);

/**
 * Gradient of iota().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Starting value.
 * @param n Length.
 * 
 * @return Gradient with respect to @p x.
 */
template<scalar T>
NUMBIRCH_KEEP Array<real,0> iota_grad(const Array<real,1>& g, const T& x, const int n);

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
template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,2> diagonal(const T& x, const int n);

/**
 * Gradient of diagonal().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 * 
 * @return Gradient with respect to @p x.
 */
template<scalar T>
NUMBIRCH_KEEP Array<real,0> diagonal_grad(const Array<real,2>& g, const T& x, const int n);

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
template<arithmetic T>
NUMBIRCH_KEEP Array<T,2> diagonal(const Array<T,1>& x);

/**
 * Gradient of diagonal().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Vector to assign to diagonal.
 * 
 * @return Gradient with respect to @p x.
 */
template<arithmetic T>
NUMBIRCH_KEEP Array<real,1> diagonal_grad(const Array<real,2>& g, const Array<T,1>& x);

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
template<arithmetic T, scalar U>
NUMBIRCH_KEEP Array<T,0> element(const Array<T,1>& x, const U& i);

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Vector.
 * @param i Index.
 * 
 * @return Gradient with respect to @p x.
 */
template<arithmetic T, scalar U>
NUMBIRCH_KEEP Array<real,1> element_grad1(const Array<real,0>& g, const Array<T,1>& x,
    const U& i);

/**
 * Gradient of element().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Vector.
 * @param i Index.
 * 
 * @return Gradient with respect to @p i.
 */
template<arithmetic T, scalar U>
NUMBIRCH_KEEP real element_grad2(const Array<real,0>& g, const Array<T,1>& x, const U& i);

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
template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP Array<T,0> element(const Array<T,2>& A, const U& i, const V& j);

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
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Gradient with respect to @p A.
 */
template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP Array<real,2> element_grad1(const Array<real,0>& g, const Array<T,2>& A,
    const U& i, const V& j);

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
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Gradient with respect to @p i.
 */
template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP real element_grad2(const Array<real,0>& g, const Array<T,2>& A, const U& i,
    const V& j);

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
 * @param A Matrix.
 * @param i Row index.
 * @param j Column index.
 * 
 * @return Gradient with respect to @p j.
 */
template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP real element_grad3(const Array<real,0>& g, const Array<T,2>& A, const U& i,
    const V& j);

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
template<scalar T, scalar U>
NUMBIRCH_KEEP Array<value_t<T>,1> single(const T& x, const U& i, const int n);

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Value of single entry.
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Gradient with respect to @p x.
 */
template<scalar T, scalar U>
NUMBIRCH_KEEP Array<real,0> single_grad1(const Array<real,1>& g, const T& x, const U& i,
    const int n);

/**
 * Gradient of single().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param x Value of single entry.
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Gradient with respect to @p i.
 */
template<scalar T, scalar U>
NUMBIRCH_KEEP real single_grad2(const Array<real,1>& g, const T& x, const U& i,
    const int n);

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
template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
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
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Gradient with respect to @p x.
 */
template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP Array<real,0> single_grad1(const Array<real,2>& g, const T& x, const U& i,
    const V& j, const int m, const int n);

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
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Gradient with respect to @p i.
 */
template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP real single_grad2(const Array<real,2>& g, const T& x, const U& i, const V& j,
    const int m, const int n);

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
 * @param x Value of single entry.
 * @param i Row of single entry (1-based).
 * @param j Column of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Gradient with respect to @p j.
 */
template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP real single_grad3(const Array<real,2>& g, const T& x, const U& i, const V& j,
    const int m, const int n);

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
template<numeric T, numeric U>
NUMBIRCH_KEEP pack_t<T,U> pack(const T& x, const U& y);

/**
 * Gradient of pack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T> pack_grad1(const real_t<pack_t<T,U>>& g, const T& x, const U& y);

/**
 * Gradient of pack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p y.
 */
template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<U> pack_grad2(const real_t<pack_t<T,U>>& g, const T& x, const U& y);

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
template<numeric T, numeric U>
NUMBIRCH_KEEP stack_t<T,U> stack(const T& x, const U& y);

/**
 * Gradient of stack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T> stack_grad1(const real_t<stack_t<T,U>>& g, const T& x, const U& y);

/**
 * Gradient of stack().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p y.
 */
template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<U> stack_grad2(const real_t<stack_t<T,U>>& g, const T& x, const U& y);

/**
 * Scalarize.
 * 
 * @ingroup array
 * 
 * @tparam T Numeric type.
 *
 * @param x Argument.
 * 
 * @return If @p x is a scalar then returns that scalar. If @p x is a vector
 * or matrix with a single element then returns that element as though a
 * slice.
 */
template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,0> scal(const T& x);

/**
 * Gradient of scal().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
NUMBIRCH_KEEP real_t<T> scal_grad(const Array<real,0>& g, const T& x);

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
template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,1> vec(const T& x);

/**
 * Gradient of vec().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
NUMBIRCH_KEEP real_t<T> vec_grad(const Array<real,1>& g, const T& x);

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
template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,2> mat(const T& x, const int n);

/**
 * Gradient of mat().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * @param n Number of columns into which to unstack.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
NUMBIRCH_KEEP real_t<T> mat_grad(const Array<real,2>& g, const T& x, const int n);

/**
 * Vector gather.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Source.
 * @param y Indices.
 * 
 * @return Result `z`, where `z[i] = x[y[i]]`.
 * 
 * @see scatter
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<T,1> gather(const Array<T,1>& x, const Array<U,1>& y);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Source.
 * @param y Indices.
 * 
 * @return Gradient with respect to @p x.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> gather_grad1(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Source.
 * @param y Indices.
 * 
 * @return Gradient with respect to @p y.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> gather_grad2(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y);

/**
 * Matrix gather.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Result `C`, where `C[i,j] = A[I[i,j], J[i,j]]`.
 * 
 * @see scatter, gather
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<T,2> gather(const Array<T,2>& A, const Array<U,2>& I,
    const Array<V,2>& J);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Gradient with respect to @p A.
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> gather_grad1(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Gradient with respect to @p I.
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> gather_grad2(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J);

/**
 * Gradient of gather().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * 
 * @return Gradient with respect to @p J.
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> gather_grad3(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J);

/**
 * Vector scatter.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
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
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<T,1> scatter(const Array<T,1>& x, const Array<U,1>& y, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Source.
 * @param y Indices.
 * @param n Length of result.
 * 
 * @return Gradient with respect to @p x.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> scatter_grad1(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Source.
 * @param y Indices.
 * @param n Length of result.
 * 
 * @return Gradient with respect to @p y.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> scatter_grad2(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y, const int n);

/**
 * Matrix scatter.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
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
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<T,2> scatter(const Array<T,2>& A, const Array<U,2>& I,
    const Array<V,2>& J, const int m, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Gradient with respect to @p A.
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> scatter_grad1(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J, const int m, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Gradient with respect to @p I.
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> scatter_grad2(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J, const int m, const int n);

/**
 * Gradient of scatter().
 * 
 * @ingroup array_grad
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param G Gradient with respect to result.
 * @param A Source.
 * @param I Row indices.
 * @param J Column indices.
 * @param m Number of rows in result.
 * @param n Number of columns in result.
 * 
 * @return Gradient with respect to @p J.
 */
template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> scatter_grad3(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J, const int m, const int n);

}
