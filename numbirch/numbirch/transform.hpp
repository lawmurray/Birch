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
 * Identity.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator+(const T& x) {
  return x;
}

/**
 * Gradient of unary operator+().
 * 
 * @ingroup transform_grad
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
default_t<T> identity_grad(const default_t<T>& g, const T& y, const T& x) {
  return g;
}

/**
 * Negation.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator-(const T& x);

/**
 * Gradient of operator!().
 * 
 * @ingroup transform_grad
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
default_t<T> negate_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  return -g;
}

/**
 * Logical `not`.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
explicit_t<bool,T> operator!(const T& x);

/**
 * Gradient of operator!().
 * 
 * @ingroup transform_grad
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
default_t<T> not_grad(const default_t<T>& g, const explicit_t<bool,T>& y,
    const T& x);

/**
 * Element-wise addition.
 * 
 * @ingroup transform
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
implicit_t<T,U> operator+(const T& x, const U& y);

/**
 * Gradient of operator+().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> add_grad(const default_t<T,U>& g,
    const implicit_t<T,U>& z, const T& x, const U& y) {
  return std::make_pair(g, g);
}

/**
 * Element-wise subtraction.
 * 
 * @ingroup transform
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
implicit_t<T,U> operator-(const T& x, const U& y);

/**
 * Gradient of operator-().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> subtract_grad(
    const default_t<T,U>& g, const implicit_t<T,U>& z, const T& x,
    const U& y) {
  return std::make_pair(g, -g);
}

/**
 * Multiplication by scalar.
 * 
 * @ingroup transform
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
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
implicit_t<T,U> operator*(const T& x, const U& y);

/**
 * Gradient of operator*().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
std::pair<default_t<T>,default_t<U>> multiply_grad(const default_t<T,U>& g,
    const implicit_t<T,U>& z, const T& x, const U& y) {
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
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * @tparam U Scalar type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
implicit_t<T,U> operator/(const T& x, const U& y);

/**
 * Gradient of operator/().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
std::pair<default_t<T>,default_t<U>> divide_grad(const default_t<T,U>& g,
    const implicit_t<T,U>& z, const T& x, const U& y) {
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

/**
 * Logical `and`.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator&&(const T& x, const U& y);

/**
 * Gradient of operator&&().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> and_grad(const default_t<T,U>& g,
    const explicit_t<bool,T,U>& z, const T& x, const U& y);

/**
 * Logical `or`.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator||(const T& x, const U& y);

/**
 * Gradient of operator||().
 * 
 * @ingroup transform_grad
 * 
* @tparam G Numeric type.
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> or_grad(const default_t<T,U>& g,
    const explicit_t<bool,T,U>& z, const T& x, const U& y);

/**
 * Element-wise equal to comparison.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator==(const T& x, const U& y);

/**
 * Gradient of operator==().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> equal_grad(const default_t<T,U>& g,
    const explicit_t<bool,T,U>& z, const T& x, const U& y);

/**
 * Element-wise not equal to comparison.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator!=(const T& x, const U& y);

/**
 * Gradient of operator!=().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> not_equal_grad(const default_t<T,U>& g,
    const explicit_t<bool,T,U>& z, const T& x, const U& y);

/**
 * Element-wise less than comparison.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator<(const T& x, const U& y);

/**
 * Gradient of operator<().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> less_grad(const default_t<T,U>& g,
    const explicit_t<bool,T,U>& z, const T& x, const U& y);

/**
 * Element-wise less than or equal to comparison.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator<=(const T& x, const U& y);

/**
 * Gradient of operator<=().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> less_or_equal_grad(
    const default_t<T,U>& g, const explicit_t<bool,T,U>& z, const T& x,
    const U& y);

/**
 * Element-wise greater than comparison.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator>(const T& x, const U& y);

/**
 * Gradient of operator>().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> greater_grad(const default_t<T,U>& g,
    const explicit_t<bool,T,U>& z, const T& x, const U& y);

/**
 * Element-wise greater than or equal to comparison.
 * 
 * @ingroup transform
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
explicit_t<bool,T,U> operator>=(const T& x, const U& y);

/**
 * Gradient of operator>=().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> greater_or_equal_grad(
    const default_t<T,U>& g, const explicit_t<bool,T,U>& z, const T& x,
    const U& y);

/**
 * Absolute value.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T abs(const T& x);

/**
 * Gradient of abs().
 * 
 * @ingroup transform_grad
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
default_t<T> abs_grad(const default_t<T>& g, const T& y, const T& x);

/**
 * Arc cosine.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> acos(const T& x);

/**
 * Gradient of acos().
 * 
 * @ingroup transform_grad
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
default_t<T> acos_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Arc sine.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> asin(const T& x);

/**
 * Gradient of asin().
 * 
 * @ingroup transform_grad
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
default_t<T> asin_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Arc tangent.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> atan(const T& x);

/**
 * Gradient of atan().
 * 
 * @ingroup transform_grad
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
default_t<T> atan_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Cast.
 * 
 * @ingroup transform
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
explicit_t<R,T> cast(const T& x);

/**
 * Gradient of cast().
 * 
 * @ingroup transform_grad
 * 
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class R, class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> cast_grad(const default_t<T>& g, const R& y, const T& x) {
  return g;
}

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T ceil(const T& x);

/**
 * Gradient of ceil().
 * 
 * @ingroup transform_grad
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
default_t<T> ceil_grad(const default_t<T>& g, const T& y, const T& x);

/**
 * Copy sign of a number.
 * 
 * @ingroup transform
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
implicit_t<T,U> copysign(const T& x, const U& y);

/**
 * Gradient of copysign().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> copysign_grad(const default_t<T,U>& g,
    const implicit_t<T,U>& z, const T& x, const U& y);

/**
 * Cosine.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> cos(const T& x);

/**
 * Gradient of cos().
 * 
 * @ingroup transform_grad
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
default_t<T> cos_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> cosh(const T& x);

/**
 * Gradient of cosh().
 * 
 * @ingroup transform_grad
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
default_t<T> cosh_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Digamma.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> digamma(const T& x);

/**
 * Multivariate digamma.
 * 
 * @ingroup transform
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
default_t<T,U> digamma(const T& x, const U& y);

/**
 * Exponential.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> exp(const T& x);

/**
 * Gradient of exp().
 * 
 * @ingroup transform_grad
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
default_t<T> exp_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  return hadamard(g, y);
}

/**
 * Exponential minus one.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> expm1(const T& x);

/**
 * Gradient of expm1().
 * 
 * @ingroup transform_grad
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
default_t<T> expm1_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  return hadamard(g, y);
}

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T floor(const T& x);

/**
 * Gradient of floor().
 * 
 * @ingroup transform_grad
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
default_t<T> floor_grad(const default_t<T>& g, const T& y, const T& x);

/**
 * Normalized lower incomplete gamma.
 * 
 * @ingroup transform
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
default_t<T,U> gamma_p(const T& x, const U& y);

/**
 * Normalized upper incomplete gamma.
 * 
 * @ingroup transform
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
default_t<T,U> gamma_q(const T& x, const U& y);

/**
 * Hadamard (element-wise) multiplication.
 * 
 * @ingroup transform
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
implicit_t<T,U> hadamard(const T& x, const U& y);

/**
 * Gradient of hadamard().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> hadamard_grad(const default_t<T,U>& g,
    const implicit_t<T,U>& z, const T& x, const U& y);

/**
 * Normalized incomplete beta.
 * 
 * @ingroup transform
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
default_t<T,U,V> ibeta(const T& x, const U& y, const V& z);

/**
 * Logarithm of beta.
 * 
 * @ingroup transform
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
default_t<T,U> lbeta(const T& x, const U& y);

/**
 * Gradient of lbeta().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> lbeta_grad(const default_t<T,U>& g,
    const default_t<T,U>& z, const T& x, const U& y);

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup transform
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
default_t<T,U> lchoose(const T& x, const U& y);

/**
 * Gradient of lchoose().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> lchoose_grad(const default_t<T,U>& g,
    const default_t<T,U>& z, const T& x, const U& y);

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> lfact(const T& x);

/**
 * Gradient of lfact().
 * 
 * @ingroup transform_grad
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
default_t<T> lfact_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Logarithm of gamma.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> lgamma(const T& x);

/**
 * Gradient of lgamma().
 * 
 * @ingroup transform_grad
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
default_t<T> lgamma_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Logarithm of multivariate gamma.
 * 
 * @ingroup transform
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
default_t<T,U> lgamma(const T& x, const U& y);

/**
 * Gradient of lgamma().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> lgamma_grad(const default_t<T,U>& g,
    const default_t<T,U>& z, const T& x, const U& y);

/**
 * Logarithm.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> log(const T& x);

/**
 * Gradient of log().
 * 
 * @ingroup transform_grad
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
default_t<T> log_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> log1p(const T& x);

/**
 * Gradient of log1p().
 * 
 * @ingroup transform_grad
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
default_t<T> log1p_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Power.
 * 
 * @ingroup transform
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
default_t<T,U> pow(const T& x, const U& y);

/**
 * Gradient of pow().
 * 
 * @ingroup transform_grad
 * 
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
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && is_compatible_v<T,U>,int>>
std::pair<default_t<T>,default_t<U>> pow_grad(const default_t<T,U>& g,
    const default_t<T,U>& z, const T& x, const U& y);

/**
 * Rectification.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T rectify(const T& x);

/**
 * Gradient of rectify().
 * 
 * @ingroup transform_grad
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
default_t<T> rectify_grad(const default_t<T>& g, const T& y, const T& x);

/**
 * Round to nearest integer value.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T round(const T& x);

/**
 * Gradient of round().
 * 
 * @ingroup transform_grad
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
default_t<T> round_grad(const default_t<T>& g, const T& y, const T& x);

/**
 * Sine.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> sin(const T& x);

/**
 * Gradient of sin().
 * 
 * @ingroup transform_grad
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
default_t<T> sin_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Hyperbolic sine.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> sinh(const T& x);

/**
 * Gradient of sinh().
 * 
 * @ingroup transform_grad
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
default_t<T> sinh_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Square root.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> sqrt(const T& x);

/**
 * Gradient of sqrt().
 * 
 * @ingroup transform_grad
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
default_t<T> sqrt_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Tangent.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> tan(const T& x);

/**
 * Gradient of tan().
 * 
 * @ingroup transform_grad
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
default_t<T> tan_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup transform
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> tanh(const T& x);

/**
 * Gradient of tanh().
 * 
 * @ingroup transform_grad
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
default_t<T> tanh_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x);

}