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
 * @ingroup numeric
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
convert_t<R,T,U> operator+(const T& x, const U& y);

/**
 * Element-wise addition.
 * 
 * @ingroup numeric
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
promote_t<T,U> operator+(const T& x, const U& y) {
  return operator+<value_t<T>>(x, y);
}

/**
 * Element-wise subtraction.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator-(const T& x, const U& y);

/**
 * Element-wise subtraction.
 * 
 * @ingroup numeric
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
promote_t<T,U> operator-(const T& x, const U& y) {
  return operator-<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Multiplication by scalar.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator*(const T& x, const U& y);

/**
 * Multiplication by scalar.
 * 
 * @ingroup numeric
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
promote_t<T,U> operator*(const T& x, const U& y) {
  return operator*<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Division by scalar.
 * 
 * @ingroup numeric
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
    is_numeric_v<T> && is_numeric_v<U> && is_scalar_v<U>,int>>
convert_t<R,T,U> operator/(const T& x, const U& y);

/**
 * Division by scalar.
 * 
 * @ingroup numeric
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
    is_numeric_v<T> && is_numeric_v<U> && is_scalar_v<U>,int>>
promote_t<T,U> operator/(const T& x, const U& y) {
  return operator/<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Logical `and`.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator&&(const T& x, const U& y);

/**
 * Logical `and`.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator&&(const T& x, const U& y) {
  return operator&&<bool>(x, y);
}

/**
 * Logical `or`.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator||(const T& x, const U& y);

/**
 * Logical `or`.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator||(const T& x, const U& y) {
  return operator||<bool>(x, y);
}

/**
 * Element-wise equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator==(const T& x, const U& y);

/**
 * Element-wise equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator==(const T& x, const U& y) {
  return operator==<bool>(x, y);
}

/**
 * Element-wise not equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator!=(const T& x, const U& y);

/**
 * Element-wise not equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator!=(const T& x, const U& y) {
  return operator!=<bool>(x, y);
}

/**
 * Element-wise less than comparison.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator<(const T& x, const U& y);

/**
 * Element-wise less than comparison.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator<(const T& x, const U& y) {
  return operator< <bool>(x, y);
  // ^ preserve the space after operator<, needed for successful parse
}

/**
 * Element-wise less than or equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator<=(const T& x, const U& y);

/**
 * Element-wise less than or equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator<=(const T& x, const U& y) {
  return operator<=<bool>(x, y);
}

/**
 * Element-wise greater than comparison.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator>(const T& x, const U& y);

/**
 * Element-wise greater than comparison.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator>(const T& x, const U& y) {
  return operator><bool>(x, y);
}

/**
 * Element-wise greater than or equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> operator>=(const T& x, const U& y);

/**
 * Element-wise greater than or equal to comparison.
 * 
 * @ingroup numeric
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
convert_t<bool,T,U> operator>=(const T& x, const U& y) {
  return operator>=<bool>(x, y);
}

/**
 * Copy sign of a number.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> copysign(const T& x, const U& y);

/**
 * Copy sign of a number.
 * 
 * @ingroup numeric
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
promote_t<T,U> copysign(const T& x, const U& y) {
  return copysign<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Multivariate digamma.
 * 
 * @ingroup numeric
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class U, class = std::enable_if_t<
    is_floating_point_v<R> &&
    is_numeric_v<T> && is_numeric_v<U>,int>>
convert_t<R,T,U> digamma(const T& x , const U& y);

/**
 * Multivariate digamma.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<
    is_numeric_v<T> && is_numeric_v<U>,int>>
promote_t<T,U> digamma(const T& x, const U& y) {
  return digamma<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Normalized lower incomplete gamma.
 * 
 * @ingroup numeric
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
    is_floating_point_v<R> && is_numeric_v<T> && is_numeric_v<U>,int>>
convert_t<R,T,U> gamma_p(const T& x, const U& y);

/**
 * Normalized lower incomplete gamma.
 * 
 * @ingroup numeric
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
    is_numeric_v<U>,int>>
promote_t<T,U> gamma_p(const T& x, const U& y) {
  return gamma_p<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Normalized upper incomplete gamma.
 * 
 * @ingroup numeric
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
    is_floating_point_v<R> && is_numeric_v<T> && is_numeric_v<U>,int>>
convert_t<R,T,U> gamma_q(const T& x, const U& y);

/**
 * Normalized upper incomplete gamma.
 * 
 * @ingroup numeric
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
    is_numeric_v<U>,int>>
promote_t<T,U> gamma_q(const T& x, const U& y) {
  return gamma_q<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Hadamard (element-wise) multiplication.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> hadamard(const T& x, const U& y);

/**
 * Hadamard (element-wise) multiplication.
 * 
 * @ingroup numeric
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
promote_t<T,U> hadamard(const T& x, const U& y) {
  return hadamard<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Logarithm of beta.
 * 
 * @ingroup numeric
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
    is_floating_point_v<R> && is_numeric_v<T> && is_numeric_v<U>,int>>
convert_t<R,T,U> lbeta(const T& x, const U& y);

/**
 * Logarithm of beta.
 * 
 * @ingroup numeric
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
promote_t<T,U> lbeta(const T& x, const U& y) {
  return lbeta<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> lchoose(const T& x, const U& y);

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup numeric
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
promote_t<T,U> lchoose(const T& x, const U& y) {
  return lchoose<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Gradient of lchoose().
 * 
 * @ingroup numeric
 * 
 * @tparam G Numeric type.
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class G, class T, class U, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<T> && is_compatible_v<G,T> &&
    is_numeric_v<U> && is_compatible_v<G,U>,int>>
std::pair<promote_t<G,T,U>,promote_t<G,T,U>> lchoose_grad(const G& g, const T& x,
    const U& y);

/**
 * Logarithm of multivariate gamma.
 * 
 * @ingroup numeric
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
convert_t<R,T,U> lgamma(const T& x, const U& y);

/**
 * Logarithm of multivariate gamma.
 * 
 * @ingroup numeric
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
promote_t<T,U> lgamma(const T& x, const U& y) {
  return lgamma<value_t<promote_t<T,U>>>(x, y);
}

/**
 * Power.
 * 
 * @ingroup numeric
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
    is_floating_point_v<R> && is_numeric_v<T> && is_numeric_v<U>,int>>
convert_t<R,T,U> pow(const T& x, const U& y);

/**
 * Power.
 * 
 * @ingroup numeric
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
promote_t<T,U> pow(const T& x, const U& y) {
  return pow<value_t<promote_t<T,U>>>(x, y);
}

}
