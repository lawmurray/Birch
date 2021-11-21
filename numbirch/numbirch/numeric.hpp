/**
 * ile
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"

namespace numbirch {
/**
 * Matrix-vector multiplication. Computes $y = Ax$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Result $y$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x);

/**
 * Matrix-matrix multiplication. Computes $C = AB$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& B);

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
 * @param A Matrix.
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
 * Computes $y = Lx$, where $S = LL^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix $S$.
 * @param x Vector $x$.
 * 
 * @return Result $y$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> cholmul(const Array<T,2>& S, const Array<T,1>& x);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a matrix.
 * Computes $C = LB$, where $S = LL^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix $S$.
 * @param B Matrix $B$.
 * 
 * @return Result $C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholmul(const Array<T,2>& S, const Array<T,2>& B);

/**
 * Outer product of matrix and lower-triangular Cholesky factor of another
 * matrix. Computes $C = AL^\top$, where $S = LL^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param S Symmetric positive definite matrix $S$.
 * 
 * @return Result $C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholouter(const Array<T,2>& A, const Array<T,2>& S);

/**
 * Matrix-vector solve, via the Cholesky factorization. Solves for $x$ in
 * $Sx = y$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix $S$.
 * @param y Vector $y$.
 * 
 * @return Result $x$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> cholsolve(const Array<T,2>& S, const Array<T,1>& y);

/**
 * Matrix-matrix solve, via the Cholesky factorization. Solves for $B$ in
 * $SB = C$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix $S$.
 * @param C Matrix $C$.
 * 
 * @return Result $B$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& C);

/**
 * Vector-vector dot product. Computes $x^\top y$, resulting in a scalar.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Dot product.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> dot(const Array<T,1>& x, const Array<T,1>& y);

/**
 * Matrix-matrix Frobenius product. Computes $\langle A, B 
 * \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$,
 * resulting in a scalar.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Frobenius product.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> frobenius(const Array<T,2>& A, const Array<T,2>& B);

/**
 * Matrix-vector inner product. Computes $y = A^\top x$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Result $y$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x);

/**
 * Matrix-matrix inner product. Computes $y = A^\top B$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B);

/**
 * Vector-vector outer product. Computes $A = xy^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $A$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,1>& x, const Array<T,1>& y);

/**
 * Matrix-matrix outer product. Computes $C = AB^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B);

/**
 * Matrix-vector solve. Solves for $x$ in $Ax = y$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param y Vector $y$.
 * 
 * @return Result $x$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> solve(const Array<T,2>& A, const Array<T,1>& y);

/**
 * Matrix-matrix solve. Solves for $B$ in $AB = C$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param C Matrix $C$.
 * 
 * @return Result $B$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> solve(const Array<T,2>& A, const Array<T,2>& C);

}
