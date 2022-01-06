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
 * @tparam G Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class T, class = std::enable_if_t<is_scalar_v<G> &&
    is_numeric_v<T>,int>>
default_t<G,T> count_grad(const G& g, const T& x);

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
 * @tparam G Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class T, class = std::enable_if_t<is_scalar_v<G> &&
    is_numeric_v<T>,int>>
default_t<G,T> sum_grad(const G& g, const T& x);

/**
 * Matrix trace.
 * 
 * @ingroup la
 * 
 * @tparam R Arithmetic type.
 * @tparam T Arithmetic type.
 * 
 * @param A Matrix.
 * 
 * @return Trace.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_arithmetic_v<T>,int>>
Array<R,0> trace(const Array<T,2>& A) {
  assert(rows(A) == columns(A));
  return sum<R>(A.diagonal());
}

/**
 * Matrix trace.
 * 
 * @ingroup la
 * 
 * @tparam T Arithmetic type.
 * 
 * @param A Matrix.
 * 
 * @return Trace.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,0> trace(const Array<T,2>& A) {
  return trace<T,T,int>(A);
}

/**
 * Gradient of trace().
 * 
 * @ingroup la
 * 
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param A Matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<T,2> trace_grad(const Array<T,0>& g, const Array<T,2>& A) {
  return diagonal(g, rows(A));
}

}
