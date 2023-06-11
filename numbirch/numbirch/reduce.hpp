/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"

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
template<numeric T>
Array<int,0> count(const T& x);

/**
 * Gradient of count().
 * 
 * @ingroup reduce_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
real_t<T> count_grad(const Array<real,0>& g, const Array<int,0>& y,
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
template<numeric T>
Array<value_t<T>,0> sum(const T& x);

/**
 * Gradient of sum().
 * 
 * @ingroup reduce_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
real_t<T> sum_grad(const Array<real,0>& g, const Array<value_t<T>,0>& y,
    const T& x);

/**
 * Minimum element.
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result; zero for empty @p x.
 */
template<numeric T>
Array<value_t<T>,0> min(const T& x);

/**
 * Gradient of min().
 * 
 * @ingroup reduce_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
real_t<T> min_grad(const Array<real,0>& g, const Array<value_t<T>,0>& y,
    const T& x);

/**
 * Maximum element.
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result; zero for empty @p x.
 */
template<numeric T>
Array<value_t<T>,0> max(const T& x);

/**
 * Gradient of max().
 * 
 * @ingroup reduce_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
real_t<T> max_grad(const Array<real,0>& g, const Array<value_t<T>,0>& y,
    const T& x);

/**
 * Cumulative sum (inclusive).
 * 
 * @ingroup reduce
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<numeric T>
T cumsum(const T& x);

/**
 * Gradient of cumsum().
 * 
 * @ingroup reduce_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<numeric T>
real_t<T> cumsum_grad(const real_t<T>& g, const T& y, const T& x);

}
