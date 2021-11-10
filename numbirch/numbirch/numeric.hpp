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
 * template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
 *    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
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
 * #is_arithmetic           | `{T: is_scalar<T> or is_array<T>}`                             |
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
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
T operator+(const T& x);

/**
 * Negation.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
T operator-(const T& x);

/**
 * Element-wise addition.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
promote_t<T,U> operator+(const T& x, const U& y);

/**
 * Element-wise subtraction.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
promote_t<T,U> operator-(const T& x, const U& y);

/**
 * Multiplication by scalar.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && (is_scalar_v<T> || is_scalar_v<U>) &&
    (!is_basic_v<T> || !is_basic_v<U>),int>>
promote_t<T,U> operator*(const T& x, const U& y);

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
 * 
 * @note @p A and @p x must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p A and @p B must have the same floating point value type `T`; this
 * is for backend compatibility.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& x);

/**
 * Division by scalar.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_scalar_v<U> && (!is_basic_v<T> || !is_basic_v<U>),int>>
promote_t<T,U> operator/(const T& x, const U& y);

/**
 * Logical `not`.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
T operator!(const T& x);

/**
 * Logical `and`.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator&&(const T& x, const U& y);

/**
 * Logical `or`.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator||(const T& x, const U& y);

/**
 * Element-wise equal to comparison.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator==(const T& x, const U& y);

/**
 * Element-wise not equal to comparison.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator!=(const T& x, const U& y);

/**
 * Element-wise less than comparison.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator<(const T& x, const U& y);

/**
 * Element-wise less than or equal to comparison.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator<=(const T& x, const U& y);

/**
 * Element-wise greater than comparison.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator>(const T& x, const U& y);

/**
 * Element-wise greater than or equal to comparison.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<bool,dimension_v<T>> operator>=(const T& x, const U& y);

/**
 * Absolute value.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
T abs(const T& x);

/**
 * Arc cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T acos(const T& x);

/**
 * Arc cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> acos(const U& x);

/**
 * Arc sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T asin(const T& x);

/**
 * Arc sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> asin(const U& x);

/**
 * Arc tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T atan(const T& x);

/**
 * Arc tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> atan(const U& x);

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>
    && !is_basic_v<T>,int>>
T ceil(const T& x);

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
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T cos(const T& x);

/**
 * Cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> cos(const U& x);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T cosh(const T& x);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> cosh(const U& x);

/**
 * Count of non-zero elements.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<int,0> count(const T& x);

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<value_t<T>,2> diagonal(const T& x, const int n);

/**
 * Digamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T digamma(const T& x);

/**
 * Digamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> digamma(const U& x);

/**
 * Exponential.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T exp(const T& x);

/**
 * Exponential.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> exp(const U& x);

/**
 * Exponential minus one.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T expm1(const T& x);

/**
 * Exponential minus one.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> expm1(const U& x);

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>
    && !is_basic_v<T>,int>>
T floor(const T& x);

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
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T lfact(const T& x);

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> lfact(const U& x);

/**
 * Gradient of lfact().
 * 
 * @ingroup numeric
 * 
 * @tparam G Floating point type.
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class T, class = std::enable_if_t<is_floating_point_v<G> &&
    is_arithmetic_v<T> && is_compatible_v<G,T>,int>>
promote_t<G,T> lfact_grad(const G& g, const T& x);

/**
 * Logarithm of the gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T lgamma(const T& x);

/**
 * Logarithm of the gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> lgamma(const U& x);

/**
 * Logarithm.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T log(const T& x);

/**
 * Logarithm.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> log(const U& x);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T log1p(const T& x);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> log1p(const U& x);

/**
 * Reciprocal. For element @f$(i,j)@f$, computes @f$B_{ij} = 1/A_{ij}@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T rcp(const T& x);

/**
 * Reciprocal. For element @f$(i,j)@f$, computes @f$B_{ij} = 1/A_{ij}@f$.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> rcp(const U& x);

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_compatible_v<T,T>,int>>
T rectify(const T& x);

/**
 * Gradient of rectify().
 * 
 * @ingroup numeric
 * 
 * @tparam G Floating point type.
 * @tparam T Arithmetic type.
 * 
 * @param g Gradient with respect to result.
 * @param x Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class G, class T, class = std::enable_if_t<is_floating_point_v<G> &&
    is_arithmetic_v<T> && is_compatible_v<G,T>,int>>
promote_t<G,T> rectify_grad(const G& g, const T& x);

/**
 * Round to nearest integer value.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>
    && !is_basic_v<T>,int>>
T round(const T& x);

/**
 * Sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T sin(const T& x);

/**
 * Sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> sin(const U& x);

/**
 * Construct single-entry vector. One of the elements of the vector is one,
 * all others are zero.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Integral type.
 * 
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * 
 * @return Single-entry vector.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_integral_v<U>,int>>
Array<T,1> single(const U& i, const int n);

/**
 * Construct single-entry matrix. One of the elements of the matrix is one,
 * all others are zero.
 * 
 * @tparam T Arithmetic type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param i Row index of single entry (1-based).
 * @param j Column index of single entry (1-based).
 * @param m Number of rows.
 * @param n Number of columns.
 * 
 * @return Single-entry matrix.
 * 
 * @note The return type `T` must be explicitly specified.
*/
template<class T, class U, class V, class = std::enable_if_t<
    is_arithmetic_v<T> && is_integral_v<U> && is_integral_v<V>,int>>
Array<T,2> single(const U& i, const V& j, const int m, const int n);

/**
 * Hyperbolic sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T sinh(const T& x);

/**
 * Hyperbolic sine.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> sinh(const U& x);

/**
 * Square root.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T sqrt(const T& x);

/**
 * Square root.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> sqrt(const U& x);

/**
 * Sum of elements.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
Array<value_t<T>,0> sum(const T& x);

/**
 * Tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T tan(const T& x);

/**
 * Tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> tan(const U& x);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T> &&
    is_compatible_v<T,T>,int>>
T tanh(const T& x);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point basic type.
 * @tparam U Integral type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument type is integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class = std::enable_if_t<is_floating_point_v<T> &&
   is_integral_v<U> && is_compatible_v<U,U>,int>>
promote_t<T,U> tanh(const U& x);

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
 * 
 * @note @p S and @p x must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p S and @p B must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p A and @p S must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p S and @p y must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p S and @p C must have the same floating point value type `T`; this
 * is for backend compatibility.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& C);

/**
 * Copy sign of a number.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
promote_t<T,U> copysign(const T& x, const U& y);

/**
 * Multivariate digamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument.
 * @param y Argument. Will be rounded down to nearest integer.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> digamma(const T& x, const U& y);

/**
 * Multivariate digamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> digamma(const U& x, const V& y);

/**
 * Vector-vector dot product. Computes @f$x^\top y@f$, resulting in a scalar.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param A Vector @f$x@f$.
 * @param B Vector @f$y@f$.
 * 
 * @return Dot product.
 * 
 * @note @p x and @p y must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param A Matrix @f$A@f$.
 * @param B Matrix @f$B@f$.
 * 
 * @return Frobenius product.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
Array<value_t<promote_t<T,U>>,0> frobenius(const T& x, const U& y);

/**
 * Normalized upper incomplete gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> gamma_p(const T& x, const U& y);

/**
 * Normalized upper incomplete gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> gamma_p(const U& x, const V& y);

/**
 * Normalized upper incomplete gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> gamma_q(const T& x, const U& y);

/**
 * Normalized upper incomplete gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> gamma_q(const U& x, const V& y);

/**
 * Hadamard (element-wise) multiplication.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U>,int>>
promote_t<T,U> hadamard(const T& x, const U& y);

/**
 * Normalized incomplete beta function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * @tparam V Arithmetic type.
 * 
 * @param x Argument.
 * @param y Argument.
 * @param z Argument.
 * 
 * @return Result.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_arithmetic_v<T> && is_arithmetic_v<U> && is_arithmetic_v<V> &&
    is_compatible_v<T,U,V> && !(is_integral_v<T> && is_integral_v<U> &&
    is_integral_v<U>),int>>
promote_t<T,U,V> ibeta(const T& x, const U& y, const V& z);

/**
 * Normalized incomplete beta function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * @tparam W Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * @param z Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class W, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_integral_v<W> && is_compatible_v<U,V,W>,int>>
promote_t<T,U> ibeta(const U& x, const V& y, const W& z);

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
 * 
 * @note @p A and @p x must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p A and @p B must have the same floating point value type `T`; this
 * is for backend compatibility.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& x);

/**
 * Logarithm of the beta function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> lbeta(const T& x, const U& y);

/**
 * Logarithm of the beta function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> lbeta(const U& x, const V& y);

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument. Will be rounded down to nearest integer.
 * @param y Argument. Will be rounded down to nearest integer.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> lchoose(const T& x, const U& y);

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> lchoose(const U& x, const V& y);

// /**
//  * Gradient of lchoose().
//  * 
//  * @ingroup numeric
//  * 
//  * @tparam T Floating point type.
//  * @tparam D Number of dimensions.
//  * 
//  * @param G %Array.
//  * @param A %Array.
//  * @param B %Array.
//  * 
//  * @return Result.
//  */
// template<class T, int D, std::enable_if_t<
//     std::is_floating_point<T>::value,int> = 0>
// std::pair<Array<T,D>,Array<T,D>> lchoose_grad(const Array<T,D>& G,
//     const Array<int,D>& A, const Array<int,D>& B) {
//   assert(G.rows() == A.rows());
//   assert(G.columns() == A.columns());
//   assert(G.rows() == B.rows());
//   assert(G.columns() == B.columns());

//   Array<T,D> GA(A.shape().compact());
//   Array<T,D> GB(A.shape().compact());
//   lchoose_grad(G.width(), G.height(), G.data(), G.stride(), A.data(),
//       A.stride(), B.data(), B.stride(), GA.data(), GA.stride(), GB.data(),
//       GB.stride());
//   return std::make_pair(GA, GB);
// }

/**
 * Logarithm of the multivariate gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument.
 * @param y Argument. Will be rounded down to nearest integer.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> lgamma(const T& x, const U& y);

/**
 * Logarithm of the multivariate gamma function.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> lgamma(const U& x, const V& y);

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
 * 
 * @note @p x and @p y must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p A and @p B must have the same floating point value type `T`; this
 * is for backend compatibility.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& x);

/**
 * Power.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Floating point type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 */
template<class T, class U, class = std::enable_if_t<is_arithmetic_v<T> &&
    is_arithmetic_v<U> && is_compatible_v<T,U> &&
    !(is_integral_v<T> && is_integral_v<U>),int>>
promote_t<T,U> pow(const T& x, const U& y);

/**
 * Power.
 * 
 * @ingroup numeric
 * 
 * @tparam T Floating point type.
 * @tparam U Integral type.
 * @tparam V Integral type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note In this overload, the argument types are integral, and so the return
 * type must be explicitly specified as floating point.
 */
template<class T, class U, class V, class = std::enable_if_t<
    is_floating_point_v<T> && is_integral_v<U> && is_integral_v<V> &&
    is_compatible_v<U,V>,int>>
promote_t<T,U> pow(const U& x, const V& y);

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
 * 
 * @note @p A and @p y must have the same floating point value type `T`; this
 * is for backend compatibility.
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
 * 
 * @note @p A and @p C must have the same floating point value type `T`; this
 * is for backend compatibility.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> solve(const Array<T,2>& A, const Array<T,2>& C);

}
