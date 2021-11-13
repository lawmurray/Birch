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
 * @ingroup numeric
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
convert_t<R,T> operator+(const T& x);

/**
 * Identity.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator+(const T& x) {
  return operator+<value_t<T>>(x);
}

/**
 * Negation.
 * 
 * @ingroup numeric
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
convert_t<R,T> operator-(const T& x);

/**
 * Negation.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator-(const T& x) {
  return operator-<value_t<T>>(x);
}

/**
 * Logical `not`.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
convert_t<bool,T> operator!(const T& x) {
  return operator!<bool>(x);
}

/**
 * Logical `not`.
 * 
 * @ingroup numeric
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
convert_t<R,T> operator!(const T& x);

/**
 * Absolute value.
 * 
 * @ingroup numeric
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
convert_t<R,T> abs(const T& x);

/**
 * Absolute value.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T abs(const T& x) {
  return abs<value_t<T>>(x);
}

/**
 * Arc cosine.
 * 
 * @ingroup numeric
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
convert_t<R,T> acos(const T& x);

/**
 * Arc cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T acos(const T& x) {
  return acos<value_t<T>>(x);
}

/**
 * Arc sine.
 * 
 * @ingroup numeric
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
convert_t<R,T> asin(const T& x);

/**
 * Arc sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T asin(const T& x) {
  return asin<value_t<T>>(x);
}

/**
 * Arc tangent.
 * 
 * @ingroup numeric
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
convert_t<R,T> atan(const T& x);

/**
 * Arc tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T atan(const T& x) {
  return atan<value_t<T>>(x);
}

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup numeric
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
convert_t<R,T> ceil(const T& x);

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T ceil(const T& x) {
  return ceil<value_t<T>>(x);
}

/**
 * Cosine.
 * 
 * @ingroup numeric
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
convert_t<R,T> cos(const T& x);

/**
 * Cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T cos(const T& x) {
  return cos<value_t<T>>(x);
}

/**
 * Hyperbolic cosine.
 * 
 * @ingroup numeric
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
convert_t<R,T> cosh(const T& x);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T cosh(const T& x) {
  return cosh<value_t<T>>(x);
}

/**
 * Digamma.
 * 
 * @ingroup numeric
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
convert_t<R,T> digamma(const T& x);

/**
 * Digamma.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T digamma(const T& x) {
  return digamma<value_t<T>>(x);
}

/**
 * Exponential.
 * 
 * @ingroup numeric
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
convert_t<R,T> exp(const T& x);

/**
 * Exponential.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T exp(const T& x) {
  return exp<value_t<T>>(x);
}

/**
 * Exponential minus one.
 * 
 * @ingroup numeric
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
convert_t<R,T> expm1(const T& x);

/**
 * Exponential minus one.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T expm1(const T& x) {
  return expm1<value_t<T>>(x);
}

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup numeric
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
convert_t<R,T> floor(const T& x);

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T floor(const T& x) {
  return floor<value_t<T>>(x);
}

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup numeric
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
convert_t<R,T> lfact(const T& x);

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T lfact(const T& x) {
  return lfact<value_t<T>>(x);
}

/**
 * Gradient of lfact().
 * 
 * @ingroup numeric
 * 
 * @tparam G Floating point type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class T, class = std::enable_if_t<is_numeric_v<G> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
promote_t<G,T> lfact_grad(const G& g, const T& x);

/**
 * Logarithm of gamma.
 * 
 * @ingroup numeric
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
convert_t<R,T> lgamma(const T& x);

/**
 * Logarithm of gamma.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T lgamma(const T& x) {
  return lgamma<value_t<T>>(x);
}

/**
 * Logarithm.
 * 
 * @ingroup numeric
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
convert_t<R,T> log(const T& x);

/**
 * Logarithm.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T log(const T& x) {
  return log<value_t<T>>(x);
}

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup numeric
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
convert_t<R,T> log1p(const T& x);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T log1p(const T& x) {
  return log1p<value_t<T>>(x);
}

/**
 * Reciprocal. For element @f$(i,j)@f$, computes @f$B_{ij} = 1/A_{ij}@f$.
 * 
 * @ingroup numeric
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
convert_t<R,T> rcp(const T& x);

/**
 * Reciprocal. For element @f$(i,j)@f$, computes @f$B_{ij} = 1/A_{ij}@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T rcp(const T& x) {
  return rcp<value_t<T>>(x);
}

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 
 * @ingroup numeric
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
convert_t<R,T> rectify(const T& x);

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T rectify(const T& x) {
  return rectify<value_t<T>>(x);
}

/**
 * Gradient of rectify().
 * 
 * @ingroup numeric
 * 
 * @tparam G Numeric type.
 * @tparam T Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class T, class = std::enable_if_t<is_numeric_v<G> &&
    is_numeric_v<T> && is_compatible_v<G,T>,int>>
promote_t<G,T> rectify_grad(const G& g, const T& x);

/**
 * Round to nearest integer value.
 * 
 * @ingroup numeric
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
convert_t<R,T> round(const T& x);

/**
 * Round to nearest integer value.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T round(const T& x) {
  return round<value_t<T>>(x);
}

/**
 * Sine.
 * 
 * @ingroup numeric
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
convert_t<R,T> sin(const T& x);

/**
 * Sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T sin(const T& x) {
  return sin<value_t<T>>(x);
}

/**
 * Hyperbolic sine.
 * 
 * @ingroup numeric
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
convert_t<R,T> sinh(const T& x);

/**
 * Hyperbolic sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T sinh(const T& x) {
  return sinh<value_t<T>>(x);
}

/**
 * Square root.
 * 
 * @ingroup numeric
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
convert_t<R,T> sqrt(const T& x);

/**
 * Square root.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T sqrt(const T& x) {
  return sqrt<value_t<T>>(x);
}

/**
 * Tangent.
 * 
 * @ingroup numeric
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
convert_t<R,T> tan(const T& x);

/**
 * Tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T tan(const T& x) {
  return tan<value_t<T>>(x);
}

/**
 * Hyperbolic tangent.
 * 
 * @ingroup numeric
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
convert_t<R,T> tanh(const T& x);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T tanh(const T& x) {
  return tanh<value_t<T>>(x);
}

}
