/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"

namespace numbirch {
/**
 * Count of non-zero elements.
 * 
 * @ingroup reduce
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_numeric_v<T>,int>>
Array<R,0> count(const T& x);

/**
 * Count of non-zero elements.
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<int,0> count(const T& x) {
  return count<int,T,int>(x);
}

/**
 * Gradient of count().
 * 
 * @ingroup unary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_numeric_v<T>,int>>
default_t<T> count_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x);

/**
 * Sum of elements.
 * 
 * @ingroup reduce
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_numeric_v<T>,int>>
Array<R,0> sum(const T& x);

/**
 * Sum of elements.
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<value_t<T>,0> sum(const T& x) {
  return sum<value_t<T>,T,int>(x);
}

/**
 * Gradient of sum().
 * 
 * @ingroup unary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_numeric_v<T>,int>>
default_t<T> sum_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x);

/**
 * Matrix trace.
 * 
 * @ingroup la
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $\mathrm{Tr}(A)$.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_numeric_v<T>,int>>
Array<R,0> trace(const T& A) {
  assert(rows(A) == columns(A));
  return sum<R>(A.diagonal());
}

/**
 * Matrix trace.
 * 
 * @ingroup la
 * 
 * @tparam T Numeric type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $\mathrm{Tr}(A)$.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<value_t<T>,0> trace(const T& A) {
  return trace<value_t<T>,T,int>(A);
}

/**
 * Gradient of trace().
 * 
 * @ingroup la
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = \mathrm{Tr}(A)$..
 * @param A Matrix $A$.
 * 
 * @return Gradient with respect to @p A.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_numeric_v<T>,int>>
default_t<T> trace_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& A) {
  return diagonal(g, rows(A));
}

}
