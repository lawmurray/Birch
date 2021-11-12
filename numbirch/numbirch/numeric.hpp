/**
 * @file
 * 
 * @defgroup numeric Numeric
 * 
 * Asynchronous numerics.
 *
 * Most functions are defined with generic parameter types. The C++ idiom of
 * *SFINAE* ("Substitution Failure Is Not An Error") is used to restrict the
 * acceptable types according to type traits. Consider, for example, the
 * addition operator, which has signature:
 * 
 * ```
 * template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
 *    is_numeric_v<U> && is_compatible_v<T,U>,int>>
 * promote_t<T,U> operator+(const T& x, const U& y);
 * ```
 * 
 * The parameter types are generic: `T` and `U`, but according to the use of
 * SFINAE in the template parameter list, both `T` and `U` must be arithmetic
 * (see below) and compatible with respect to their number of dimensions. The
 * return type is inferred from `T` and `U` (see below).
 * 
 * Type traits are defined as:
 * 
 * Trait                    | Set of eligible types                                          |
 * :----------------------- | :------------------------------------------------------------- |
 * `is_boolean`             | `{bool}`                                                       |
 * `is_integral`            | `{int}`                                                        |
 * `is_floating_point`      | `{double, float}`                                              |
 * `is_scalar`              | `{T: is_boolean<T> or is_integral<T> or is_floating_point<T>}` |
 * `is_array`               | `{Array<T,D>: is_scalar<T>}`                                   |
 * #is_numeric           | `{T: is_scalar<T> or is_array<T>}`                             |
 * 
 * Where the return type is `promote<T,U>`. The numerical promotion rules are
 * defined as follows (these are symmetric, i.e. swap `T` and `U` if
 * necessary):
 * 
 * `T`                      | `U`                 | `promote<T,U>`  |
 * :----------------------- | ------------------- | --------------- |
 * `double`                 | `double`            | `double`        |
 * `double`                 | `float`             | `double`        |
 * `double`                 | `int`               | `double`        |
 * `float`                  | `float`             | `float`         |
 * `float`                  | `int`               | `float`         |
 * `int`                    | `int`               | `int`           |
 * 
 * Similarly, an operation between `Array<T,D>` and `Array<U,D>` promotes to
 * `Array<promote<T,U>,D>`.
 * 
 * These rules differ from numerical promotion in C++. In C++ an operation
 * between `f;pat` and `int` promotes to `double`. That it instead promotes to
 * `float` here is intended to mitigate accidental promotion from single- to
 * double-precision computations, which has serious performance implications
 * on many devices.
 * 
 * Finally, `is_compatible` requires that the number of dimensions of each
 * type matches. Zero-dimensional arrays are also compatible with scalars,
 * i.e. `is_compatible<Array<T,0>,U>` is true if `is_scalar<U>`. On the other
 * hand, two scalar types are never compatible, i.e. `is_compatible<T,U>` is
 * false for `is_scalar<T>` and `is_scalar<U>`: in such cases, built-in
 * operators and standard functions would typically be used instead of
 * NumBirch versions.
 * 
 * Some functions have a second overload that takes integral arguments, but
 * requires that a floating point type be explicitly provided for the return
 * type. Consider, for example, lfact (logarithm of the factorial function)
 * and lchoose (logarithm of the binomial coefficient): it makes sense for
 * these functions to accept integral arguments, but it does not make sense
 * for them to return integers (given the logarithm), so the desired floating
 * point return type must be explicitly specified when calling, e.g.
 * 
 * ```
 * int x = 100;
 * double y = lfact<double>(x);
 * ```
 * 
 * On the other hand, one can call the standard overload of these functions,
 * with a floating point argument, to avoid explicitly specifying the return
 * type:
 * 
 * ```
 * double x = 100.0;
 * double y = lfact(x);
 * ```
 * 
 * Some functions, notably those typically implemented in BLAS or LAPACK, only
 * accept matrix or vector arguments of the same floating point type (i.e. all
 * `float` or all `double`). This restriction is maintained by the NumBirch
 * interface to maximize backend compatibility.
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
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param x Vector @f$x@f$.
 * 
 * @return Result @f$y@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x);

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param B Matrix @f$B@f$.
 * 
 * @return Result @f$C@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& x);

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
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Matrix.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholinv(const Array<T,2>& S);

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
 * Count of non-zero elements.
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
Array<R,0> count(const T& x);

/**
 * Count of non-zero elements.
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
Array<int,0> count(const T& x) {
  return count<int>(x);
}

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup numeric
 * 
 * @tparam R Arithmetic type.
 * @tparam T Scalar type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_scalar_v<T>,int>>
Array<R,2> diagonal(const T& x, const int n);

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup numeric
 * 
 * @tparam T Numeric type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
Array<value_t<T>,2> diagonal(const T& x, const int n) {
  return diagonal<value_t<T>>(x, n);
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
 * Inverse of a square matrix.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Square matrix.
 * 
 * @return Inverse.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inv(const Array<T,2>& A);

/**
 * Logarithm of the determinant of a symmetric positive definite matrix, via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Logarithm of the determinant.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> lcholdet(const Array<T,2>& S);


/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Logarithm of the absolute value of the determinant.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> ldet(const Array<T,2>& A);

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
 * Construct single-entry vector. One of the elements of the vector is one,
 * all others are zero.
 * 
 * @ingroup numeric
 * 
 * @tparam R Arithmetic type.
 * @tparam T Scalar type.
 * 
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Single-entry vector.
 */
template<class R, class T, class = std::enable_if_t<is_arithmetic_v<R> &&
    is_scalar_v<T>,int>>
Array<R,1> single(const T& i, const int n);

/**
 * Construct single-entry matrix. One of the elements of the matrix is one,
 * all others are zero.
 * 
 * @tparam R Arithmetic type.
 * @tparam T Scalar type.
 * @tparam U Scalar type.
 * 
 * @param i Row index of single entry (1-based).
 * @param j Column index of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Single-entry matrix.
*/
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> && is_scalar_v<T> && is_scalar_v<U>,int>>
Array<R,2> single(const T& i, const U& j, const int m, const int n);

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
 * Sum of elements.
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
Array<R,0> sum(const T& x);

/**
 * Sum of elements.
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
Array<value_t<T>,0> sum(const T& x) {
  return sum<value_t<T>>(x);
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

/**
 * Matrix trace.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix.
 * 
 * @return Trace.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> trace(const Array<T,2>& A);

/**
 * Matrix transpose.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix.
 * 
 * @return Transpose.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> transpose(const Array<T,2>& A);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a vector.
 * Computes @f$y = Lx@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix @f$S@f$.
 * @param x Vector @f$x@f$.
 * 
 * @return Result @f$y@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> cholmul(const Array<T,2>& S, const Array<T,1>& x);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a matrix.
 * Computes @f$C = LB@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix @f$S@f$.
 * @param B Matrix @f$B@f$.
 * 
 * @return Result @f$C@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholmul(const Array<T,2>& S, const Array<T,2>& B);

/**
 * Outer product of matrix and lower-triangular Cholesky factor of another
 * matrix. Computes @f$C = AL^\top@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param S Symmetric positive definite matrix @f$S@f$.
 * 
 * @return Result @f$C@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholouter(const Array<T,2>& A, const Array<T,2>& S);

/**
 * Matrix-vector solve, via the Cholesky factorization. Solves for @f$x@f$ in
 * @f$Sx = y@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix @f$S@f$.
 * @param y Vector @f$y@f$.
 * 
 * @return Result @f$x@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> cholsolve(const Array<T,2>& S, const Array<T,1>& y);

/**
 * Matrix-matrix solve, via the Cholesky factorization. Solves for @f$B@f$ in
 * @f$SB = C@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix @f$S@f$.
 * @param C Matrix @f$C@f$.
 * 
 * @return Result @f$B@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& C);

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
 * Vector-vector dot product. Computes @f$x^\top y@f$, resulting in a scalar.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Vector @f$x@f$.
 * @param B Vector @f$y@f$.
 * 
 * @return Dot product.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> dot(const Array<T,1>& x, const Array<T,1>& y);

/**
 * Matrix-matrix Frobenius product. Computes @f$\langle A, B 
 * \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}@f$,
 * resulting in a scalar.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param B Matrix @f$B@f$.
 * 
 * @return Frobenius product.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> frobenius(const Array<T,2>& x, const Array<T,2>& y);

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
 * Normalized incomplete beta.
 * 
 * @ingroup numeric
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
    is_floating_point_v<R> && is_numeric_v<T> && is_numeric_v<U> &&
    is_numeric_v<V> && is_compatible_v<T,U,V>,int>>
convert_t<R,T,U,V> ibeta(const T& x, const U& y, const V& z);

/**
 * Normalized incomplete beta.
 * 
 * @ingroup numeric
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
promote_t<T,U,V> ibeta(const T& x, const U& y, const V& z) {
  return ibeta<value_t<promote_t<T,U,V>>>(x, y, z);
}

/**
 * Matrix-vector inner product. Computes @f$y = A^\top x@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param x Vector @f$x@f$.
 * 
 * @return Result @f$y@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x);

/**
 * Matrix-matrix inner product. Computes @f$y = A^\top x@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param B Matrix @f$B@f$.
 * 
 * @return Result @f$C@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& x);

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
 * Vector-vector outer product. Computes @f$A = xy^\top@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector @f$x@f$.
 * @param y Vector @f$y@f$.
 * 
 * @return Result @f$A@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,1>& x, const Array<T,1>& y);

/**
 * Matrix-matrix outer product. Computes @f$C = AB^\top@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param B Matrix @f$B@f$.
 * 
 * @return Result @f$C@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& x);

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

/**
 * Matrix-vector solve. Solves for @f$x@f$ in @f$Ax = y@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param y Vector @f$y@f$.
 * 
 * @return Result @f$x@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> solve(const Array<T,2>& A, const Array<T,1>& y);

/**
 * Matrix-matrix solve. Solves for @f$B@f$ in @f$AB = C@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix @f$A@f$.
 * @param C Matrix @f$C@f$.
 * 
 * @return Result @f$B@f$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> solve(const Array<T,2>& A, const Array<T,2>& C);

}
