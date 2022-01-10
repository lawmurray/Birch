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
 * Gradient of operator*().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p A and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> multiply_grad(const Array<T,1>& g,
    const Array<T,2>& A, const Array<T,1>& x) {
  return std::make_pair(outer(g, x), inner(A, g));
}

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
 * Gradient of operator*().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> multiply_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(outer(G, B), inner(A, G));
}

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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholinv(const Array<T,2>& S);

/**
 * Gradient of cholinv().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param S Symmetric positive definite matrix.
 * 
 * @return Gradient with respect to @p S.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholinv_grad(const Array<T,2>& G, const Array<T,2>& S) {
  auto B = cholinv(S);
  return -B*G*B;
}

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
 * Gradient of inv().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Square matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inv_grad(const Array<T,2>& G, const Array<T,2>& A) {
  auto B = inv(A);
  return -outer(inner(B, G), B);
}

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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> lcholdet(const Array<T,2>& S);

/**
 * Gradient of `lcholdet()`.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param S Symmetric positive definite matrix.
 * 
 * @return Gradient with respect to @p S.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> lcholdet_grad(const Array<T,0>& g, const Array<T,2>& S) {
  return g*cholinv(S);
}

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
 * Gradient of `ldet()`.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param A Matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> ldet_grad(const Array<T,0>& g, const Array<T,2>& A) {
  return g*transpose(inv(A));
}

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
 * Gradient of transpose().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> transpose_grad(const Array<T,2>& G, const Array<T,2>& A) {
  return transpose(G);
}

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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> cholsolve(const Array<T,2>& S, const Array<T,1>& y);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param S Symmetric positive definite matrix $S$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p S and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> cholsolve_grad(const Array<T,1>& g,
    const Array<T,2>& S, const Array<T,1>& y) {
  auto L = cholinv(S);
  return std::make_pair(-L*outer(g, y)*L, inner(L, g));
}

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
 * 
 * @note Backends may implement this in various ways to improve robustness,
 * e.g. using an $LDL^\top$ (Bunch-Kaufman) factorization, or retrying a
 * failed single precision factorization in double precision.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& C);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param S Symmetric positive definite matrix $S$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p S and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> cholsolve_grad(const Array<T,2>& G,
    const Array<T,2>& S, const Array<T,2>& C) {
  auto L = cholinv(S);
  return std::make_pair(-L*outer(G, C)*L, inner(L, G));
}

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
 * Gradient of dot().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,1>,Array<T,1>> dot_grad(const Array<T,0>& g,
    const Array<T,1>& x, const Array<T,1>& y) {
  return std::make_pair(g*y, g*x);
}

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
 * Gradient of frobenius().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> frobenius_grad(const Array<T,0>& g,
    const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(g*B, g*A);
}

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
 * Gradient of inner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p A and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> inner_grad(const Array<T,1>& g,
    const Array<T,2>& A, const Array<T,1>& x) {
  return std::make_pair(outer(x, g), A*g);
}

/**
 * Matrix-vector inner product and addition. Computes $z = x + A^\top y$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * @param A Matrix $A$.
 * @param y Vector $y$.
 * 
 * @return Result $z$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> inner(const Array<T,1>& x, const Array<T,2>& A,
    const Array<T,1>& y);

/**
 * Gradient of inner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param x Vector $x$.
 * @param A Matrix $A$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p x, @p A and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,1>,Array<T,2>,Array<T,1>> inner_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& A, const Array<T,1>& y) {
  return std::make_tuple(g, outer(y, g), A*g);
}

/**
 * Matrix-matrix inner product. Computes $C = A^\top B$.
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
 * Gradient of inner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> inner_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(outer(B, G), A*G);
}

/**
 * Matrix-matrix inner product and addition. Computes $D = A + B^\top C$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Result $D$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B,
    const Array<T,2>& C);

/**
 * Gradient of inner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p A, @p B and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,2>,Array<T,2>,Array<T,2>> inner_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,2>& B, const Array<T,2>& C) {
  return std::make_tuple(G, outer(C, G), B*G);
}

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
 * Gradient of outer().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,1>,Array<T,1>> outer_grad(const Array<T,2>& G,
    const Array<T,1>& x, const Array<T,1>& y) {
  return std::make_pair(G*y, inner(G, x));
}

/**
 * Vector-vector outer product and addition. Computes $B = A + xy^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $B$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,2>& A, const Array<T,1>& x,
    const Array<T,1>& y);

/**
 * Gradient of outer().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p A, @p x and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,2>,Array<T,1>,Array<T,1>> outer_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,1>& x, const Array<T,1>& y) {
  return std::make_tuple(G, G*y, inner(G, x));
}

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
 * Gradient of outer().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> outer_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(G*B, inner(G, A));
}

/**
 * Matrix-matrix outer product and addition. Computes $D = A + BC^\top$.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Result $D$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B,
    const Array<T,2>& C);

/**
 * Gradient of outer().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p A, @p B and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,2>,Array<T,2>,Array<T,2>> outer_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,2>& B, const Array<T,2>& C) {
  return std::make_tuple(G, G*C, inner(G, B));
}

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
 * Gradient of solve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param A Matrix $A$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p A and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> solve_grad(const Array<T,1>& g,
    const Array<T,2>& A, const Array<T,1>& y) {
  auto L = inv(A);
  return std::make_pair(-L*outer(g, y)*L, inner(L, g));
}

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

/**
 * Gradient of cholsolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param G Gradient with respect to result.
 * @param A Matrix $A$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p A and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> solve_grad(const Array<T,2>& G,
    const Array<T,2>& A, const Array<T,2>& C) {
  auto L = inv(A);
  return std::make_pair(-L*outer(G, C)*L, inner(L, G));
}

}
