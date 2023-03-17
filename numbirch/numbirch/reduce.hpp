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
 * Count non-zero elements.
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result; zero for empty @p x.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<int,0> count(const T& x);

/**
 * Gradient of count().
 * 
 * @ingroup reduce_grad
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
real_t<T> count_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x);

/**
 * Sum elements.
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result; zero for empty @p x.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
Array<value_t<T>,0> sum(const T& x);

/**
 * Gradient of sum().
 * 
 * @ingroup reduce_grad
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
real_t<T> sum_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x);

}
