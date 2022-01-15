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
 * @ingroup unary
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
explicit_t<R,T> operator+(const T& x);

/**
 * Identity.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator+(const T& x) {
  return operator+<value_t<T>,T,int>(x);
}

/**
 * Gradient of unary operator+().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
G identity_grad(const G& g, const T& y, const T& x) {
  return g;
}

/**
 * Negation.
 * 
 * @ingroup unary
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
explicit_t<R,T> operator-(const T& x);

/**
 * Negation.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator-(const T& x) {
  return operator-<value_t<T>,T,int>(x);
}

/**
 * Gradient of operator!().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
G negate_grad(const G& g, const R& y, const T& x) {
  return -g;
}

/**
 * Logical `not`.
 * 
 * @ingroup unary
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
explicit_t<R,T> operator!(const T& x);

/**
 * Logical `not`.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
explicit_t<bool,T> operator!(const T& x) {
  return operator!<bool,T,int>(x);
}

/**
 * Gradient of operator!().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> not_grad(const G& g, const R& y, const T& x);

/**
 * Absolute value.
 * 
 * @ingroup unary
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
explicit_t<R,T> abs(const T& x);

/**
 * Absolute value.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T abs(const T& x) {
  return abs<value_t<T>,T,int>(x);
}

/**
 * Gradient of abs().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> abs_grad(const G& g, const R& y, const T& x);

/**
 * Arc cosine.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> acos(const T& x);

/**
 * Arc cosine.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> acos(const T& x) {
  return acos<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of acos().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> acos_grad(const G& g, const R& y, const T& x);

/**
 * Arc sine.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> asin(const T& x);

/**
 * Arc sine.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> asin(const T& x) {
  return asin<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of asin().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> asin_grad(const G& g, const R& y, const T& x);

/**
 * Arc tangent.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> atan(const T& x);

/**
 * Arc tangent.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> atan(const T& x) {
  return atan<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of atan().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> atan_grad(const G& g, const R& y, const T& x);

/**
 * Cast.
 * 
 * @ingroup unary
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
 * Cast.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T cast(const T& x) {
  return cast<value_t<T>,T,int>(x);
}

/**
 * Gradient of cast().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
G cast_grad(const G& g, const R& y, const T& x) {
  return g;
}

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup unary
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
explicit_t<R,T> ceil(const T& x);

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T ceil(const T& x) {
  return ceil<value_t<T>,T,int>(x);
}

/**
 * Gradient of ceil().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> ceil_grad(const G& g, const R& y, const T& x);

/**
 * Cosine.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> cos(const T& x);

/**
 * Cosine.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> cos(const T& x) {
  return cos<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of cos().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> cos_grad(const G& g, const R& y, const T& x);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> cosh(const T& x);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> cosh(const T& x) {
  return cosh<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of cosh().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> cosh_grad(const G& g, const R& y, const T& x);

/**
 * Digamma.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> digamma(const T& x);

/**
 * Digamma.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> digamma(const T& x) {
  return digamma<value_t<default_t<T>>,T,int>(x);
}

/**
 * Exponential.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> exp(const T& x);

/**
 * Exponential.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> exp(const T& x) {
  return exp<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of exp().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> exp_grad(const G& g, const R& y, const T& x);

/**
 * Exponential minus one.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> expm1(const T& x);

/**
 * Exponential minus one.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> expm1(const T& x) {
  return expm1<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of expm1().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> expm1_grad(const G& g, const R& y, const T& x);

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup unary
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
explicit_t<R,T> floor(const T& x);

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T floor(const T& x) {
  return floor<value_t<T>,T,int>(x);
}

/**
 * Gradient of floor().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> floor_grad(const G& g, const R& y, const T& x);

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> lfact(const T& x);

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> lfact(const T& x) {
  return lfact<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of lfact().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> lfact_grad(const G& g, const R& y, const T& x);

/**
 * Logarithm of gamma.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
    is_numeric_v<T>,int>>
explicit_t<R,T> lgamma(const T& x);

/**
 * Logarithm of gamma.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> lgamma(const T& x) {
  return lgamma<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of lgamma().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> lgamma_grad(const G& g, const R& y, const T& x);

/**
 * Logarithm.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> log(const T& x);

/**
 * Logarithm.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> log(const T& x) {
  return log<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of log().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> log_grad(const G& g, const R& y, const T& x);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> log1p(const T& x);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> log1p(const T& x) {
  return log1p<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of log1p().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> log1p_grad(const G& g, const R& y, const T& x);

/**
 * Rectification.
 * 
 * @ingroup unary
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
explicit_t<R,T> rectify(const T& x);

/**
 * Rectification.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T rectify(const T& x) {
  return rectify<value_t<T>,T,int>(x);
}

/**
 * Gradient of rectify().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> rectify_grad(const G& g, const R& y, const T& x);

/**
 * Round to nearest integer value.
 * 
 * @ingroup unary
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
explicit_t<R,T> round(const T& x);

/**
 * Round to nearest integer value.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T round(const T& x) {
  return round<value_t<T>,T,int>(x);
}

/**
 * Gradient of round().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> round_grad(const G& g, const R& y, const T& x);

/**
 * Sine.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> sin(const T& x);

/**
 * Sine.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> sin(const T& x) {
  return sin<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of sin().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> sin_grad(const G& g, const R& y, const T& x);

/**
 * Hyperbolic sine.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> sinh(const T& x);

/**
 * Hyperbolic sine.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> sinh(const T& x) {
  return sinh<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of sinh().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> sinh_grad(const G& g, const R& y, const T& x);

/**
 * Square root.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> sqrt(const T& x);

/**
 * Square root.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> sqrt(const T& x) {
  return sqrt<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of sqrt().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> sqrt_grad(const G& g, const R& y, const T& x);

/**
 * Tangent.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> tan(const T& x);

/**
 * Tangent.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> tan(const T& x) {
  return tan<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of tan().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> tan_grad(const G& g, const R& y, const T& x);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup unary
 * 
 * @tparam R Floating point type.
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class R, class T, class = std::enable_if_t<is_floating_point_v<R> &&
   is_numeric_v<T>,int>>
explicit_t<R,T> tanh(const T& x);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup unary
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
default_t<T> tanh(const T& x) {
  return tanh<value_t<default_t<T>>,T,int>(x);
}

/**
 * Gradient of tanh().
 * 
 * @ingroup unary
 * 
 * @tparam G Numeric type.
 * @tparam R Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class R, class T, class = std::enable_if_t<
    is_numeric_v<G> &&
    is_numeric_v<R> && is_compatible_v<G,R> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
default_t<G,T> tanh_grad(const G& g, const R& y, const T& x);

}
