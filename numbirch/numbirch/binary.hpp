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
 * Element-wise addition.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> && is_numeric_v<T> && is_numeric_v<U> &&
    is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator+(const T& x, const U& y);

/**
 * Element-wise addition.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
implicit_t<T,U> operator+(const T& x, const U& y) {
  return operator+<value_t<implicit_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of operator+().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T,U>,default_t<T,U>> add_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y) {
  return std::make_pair(g, g);
}

/**
 * Element-wise subtraction.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator-(const T& x, const U& y);

/**
 * Element-wise subtraction.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
implicit_t<T,U> operator-(const T& x, const U& y) {
  return operator-<value_t<implicit_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of operator-().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T,U>,default_t<T,U>> subtract_grad(
    const default_t<T,U>& g, const explicit_t<R,T,U>& z, const T& x,
    const U& y) {
  return std::make_pair(g, -g);
}

/**
 * Logical `and`.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator&&(const T& x, const U& y);

/**
 * Logical `and`.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator&&(const T& x, const U& y) {
  return operator&&<bool,T,U,int>(x, y);
}

/**
 * Gradient of operator&&().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> and_grad(const default_t<T,U>& g, const explicit_t<R,T,U>& z,
    const T& x, const U& y);

/**
 * Logical `or`.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator||(const T& x, const U& y);

/**
 * Logical `or`.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator||(const T& x, const U& y) {
  return operator||<bool,T,U,int>(x, y);
}

/**
 * Gradient of operator||().
 * 
 * @ingroup binary
 * 
* @tparam G Numeric type.
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> or_grad(const default_t<T,U>& g, const explicit_t<R,T,U>& z,
    const T& x, const U& y);

/**
 * Element-wise equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator==(const T& x, const U& y);

/**
 * Element-wise equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator==(const T& x, const U& y) {
  return operator==<bool,T,U,int>(x, y);
}

/**
 * Gradient of operator==().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> equal_grad(
    const default_t<T,U>& g, const explicit_t<R,T,U>& z, const T& x,
    const U& y);

/**
 * Element-wise not equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator!=(const T& x, const U& y);

/**
 * Element-wise not equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator!=(const T& x, const U& y) {
  return operator!=<bool,T,U,int>(x, y);
}

/**
 * Gradient of operator!=().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> not_equal_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Element-wise less than comparison.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator<(const T& x, const U& y);

/**
 * Element-wise less than comparison.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator<(const T& x, const U& y) {
  return operator< <bool,T,U,int>(x, y);
  // ^ preserve the space after operator<, needed for successful parse
}

/**
 * Gradient of operator<().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> less_grad(const default_t<T,U>& g, const explicit_t<R,T,U>& z,
    const T& x, const U& y);

/**
 * Element-wise less than or equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator<=(const T& x, const U& y);

/**
 * Element-wise less than or equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator<=(const T& x, const U& y) {
  return operator<=<bool,T,U,int>(x, y);
}

/**
 * Gradient of operator<=().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> less_or_equal_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Element-wise greater than comparison.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator>(const T& x, const U& y);

/**
 * Element-wise greater than comparison.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator>(const T& x, const U& y) {
  return operator><bool,T,U,int>(x, y);
}

/**
 * Gradient of operator>().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> greater_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Element-wise greater than or equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> operator>=(const T& x, const U& y);

/**
 * Element-wise greater than or equal to comparison.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<bool,implicit_t<T,U>> operator>=(const T& x, const U& y) {
  return operator>=<bool,T,U,int>(x, y);
}

/**
 * Gradient of operator>=().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> greater_or_equal_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Copy sign of a number.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> copysign(const T& x, const U& y);

/**
 * Copy sign of a number.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
implicit_t<T,U> copysign(const T& x, const U& y) {
  return copysign<value_t<implicit_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of copysign().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> copysign_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Multivariate digamma.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> digamma(const T& x, const U& y);

/**
 * Multivariate digamma.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> digamma(const T& x, const U& y) {
  return digamma<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Normalized lower incomplete gamma.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> gamma_p(const T& x, const U& y);

/**
 * Normalized lower incomplete gamma.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> gamma_p(const T& x, const U& y) {
  return gamma_p<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Normalized upper incomplete gamma.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> gamma_q(const T& x, const U& y);

/**
 * Normalized upper incomplete gamma.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> gamma_q(const T& x, const U& y) {
  return gamma_q<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Hadamard (element-wise) multiplication.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
explicit_t<R,T,U> hadamard(const T& x, const U& y);

/**
 * Hadamard (element-wise) multiplication.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> && is_compatible_v<T,U>,int>>
implicit_t<T,U> hadamard(const T& x, const U& y) {
  return hadamard<value_t<implicit_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of hadamard().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> hadamard_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Logarithm of beta.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> lbeta(const T& x, const U& y);

/**
 * Logarithm of beta.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> lbeta(const T& x, const U& y) {
  return lbeta<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of lbeta().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> lbeta_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> lchoose(const T& x, const U& y);

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> lchoose(const T& x, const U& y) {
  return lchoose<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of lchoose().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> lchoose_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Logarithm of multivariate gamma.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> lgamma(const T& x, const U& y);

/**
 * Logarithm of multivariate gamma.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> lgamma(const T& x, const U& y) {
  return lgamma<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of lgamma().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> lgamma_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y);

/**
 * Power.
 * 
 * @ingroup binary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
explicit_t<R,T,U> pow(const T& x, const U& y);

/**
 * Power.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
default_t<T,U> pow(const T& x, const U& y) {
  return pow<value_t<default_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of pow().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_numeric_v<U>,int>>
std::pair<default_t<T>,default_t<U>> pow_grad(const default_t<T,U>& g, const explicit_t<R,T,U>& z,
    const T& x, const U& y);

/**
 * Multiplication by scalar.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> &&
    (is_scalar_v<T> || is_scalar_v<U>),int>>
explicit_t<R,T,U> operator*(const T& x, const U& y);

/**
 * Multiplication by scalar.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U> &&
    (is_scalar_v<T> || is_scalar_v<U>),int>>
implicit_t<T,U> operator*(const T& x, const U& y) {
  return operator*<value_t<implicit_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of operator*().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_numeric_v<U> &&
    (is_scalar_v<T> || is_scalar_v<U>),int>>
std::pair<default_t<T>,default_t<U>> multiply_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y) {
  if constexpr (is_scalar_v<T> && is_scalar_v<U>) {
    return std::make_pair(g*y, g*x);
  } else if constexpr (is_scalar_v<T> && is_vector_v<U>) {
    return std::make_pair(dot(g, y), g*x);
  } else if constexpr (is_vector_v<T> && is_scalar_v<U>) {
    return std::make_pair(g*y, dot(g, x));
  } else if constexpr (is_scalar_v<T> && is_matrix_v<U>) {
    return std::make_pair(frobenius(g, y), g*x);
  } else if constexpr (is_matrix_v<T> && is_scalar_v<U>) {
    return std::make_pair(g*y, frobenius(g, x));
  } else {
    assert(false);
  }
}

/**
 * Division by scalar.
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Scalar type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> && is_scalar_v<U>,int>>
explicit_t<R,T,U> operator/(const T& x, const U& y);

/**
 * Division by scalar.
 * 
 * @ingroup binary
 * 
 * @tparam T Numeric type.
 * @tparam U Scalar type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_scalar_v<U>,int>>
implicit_t<T,U> operator/(const T& x, const U& y) {
  return operator/<value_t<implicit_t<T,U>>,T,U,int>(x, y);
}

/**
 * Gradient of operator/().
 * 
 * @ingroup binary
 * 
 * @tparam R Arithmetic type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> &&
    is_numeric_v<T> &&
    is_scalar_v<U>,int>>
std::pair<default_t<T>,default_t<U>> divide_grad(const default_t<T,U>& g,
    const explicit_t<R,T,U>& z, const T& x, const U& y) {
  if constexpr (is_scalar_v<T>) {
    return std::make_pair(g/y, -g*x/(y*y));
  } else if constexpr (is_vector_v<T>) {
    return std::make_pair(g/y, -dot(g, x)/(y*y));
  } else if constexpr (is_matrix_v<T>) {
    return std::make_pair(g/y, -frobenius(g, x)/(y*y));
  } else {
    assert(false);
  }
}

}
