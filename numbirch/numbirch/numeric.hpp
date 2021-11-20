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
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * Inverse of a square matrix.
 * 
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * Matrix trace.
 * 
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
 * @ingroup la
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
