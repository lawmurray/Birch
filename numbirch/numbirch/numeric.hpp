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
 * #is_numeric              | `{T: is_scalar<T> or is_array<T>}`                             |
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
