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
 * Normalized incomplete beta.
 * 
 * @ingroup ternary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * @tparam V Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * @param z Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_numeric_v<V> &&
    is_compatible_v<T,U,V>,int>>
explicit_t<R,T,U,V> ibeta(const T& x, const U& y, const V& z);

/**
 * Normalized incomplete beta.
 * 
 * @ingroup ternary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * @tparam V Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * @param z Argument.
 * 
 * @return Result.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_numeric_v<V> &&
    is_compatible_v<T,U,V>,int>>
default_t<T,U,V> ibeta(const T& x, const U& y, const V& z) {
  return ibeta<value_t<default_t<T,U,V>>,T,U,V,int>(x, y, z);
}

}
