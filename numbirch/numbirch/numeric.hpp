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
 * Matrix-vector multiplication.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Result $y = Ax$.
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
 * @param y Result $y = Ax$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p A and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> multiply_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& A, const Array<T,1>& x) {
  return std::make_pair(outer(g, x), inner(A, g));
}

/**
 * Matrix-matrix multiplication.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = AB$.
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
 * @param g Gradient with respect to result.
 * @param C Result $C = AB$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> multiply_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(outer(g, B), inner(A, g));
}

/**
 * Cholesky factorization of a symmetric positive definite square matrix.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix $S$.
 * 
 * @return Lower-triangular Cholesky factor $L$ such that $S = LL^\top$. If
 * the factorization fails, then $L$ is filled with NaN.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> chol(const Array<T,2>& S);

/**
 * Gradient of chol().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param L Lower-triangular Cholesky factor $L$ such that $S = LL^\top$.
 * @param S Symmetric positive definite matrix $S$.
 * 
 * @return Gradient with respect to @p S.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> chol_grad(const Array<T,2>& g, const Array<T,2>& L,
    const Array<T,2>& S);

/**
 * Inverse of a symmetric positive definite square matrix via the Cholesky
 * factorization.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Result $S^{-1} = (LL^\top)^{-1}$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholinv(const Array<T,2>& L);

/**
 * Gradient of cholinv().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = S^{-1} = (LL^\top)^{-1}$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholinv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& L);

/**
 * Matrix-vector solve via the Cholesky factorization.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Vector $y$.
 * 
 * @return Solution of $x$ in $Sx = LL^\top x = y$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> cholsolve(const Array<T,2>& L, const Array<T,1>& y);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $Sx = LL^\top x = y$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p L and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> cholsolve_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& L, const Array<T,1>& y);

/**
 * Matrix-matrix solve via the Cholesky factorization.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param C Matrix $C$.
 * 
 * @return Solution of $B$ in $SB = LL^\top B = C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> cholsolve(const Array<T,2>& L, const Array<T,2>& C);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $SB = LL^\top B = C$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p L and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> cholsolve_grad(const Array<T,2>& g,
    const Array<T,2>& B, const Array<T,2>& L, const Array<T,2>& C);

/**
 * Vector dot product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * 
 * @return Result $x^\top x$ as a scalar.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> dot(const Array<T,1>& x) {
  return dot(x, x);
}

/**
 * Gradient of dot().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = x^\top x$ as a scalar.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> dot_grad(const Array<T,0>& g, const Array<T,0>& z,
    const Array<T,1>& x) {
  return T(2.0)*g*x;
}

/**
 * Vector-vector dot product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $x^\top y$ as a scalar.
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
 * @param z Result $z = x^\top y$ as a scalar.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,1>,Array<T,1>> dot_grad(const Array<T,0>& g,
    const Array<T,0>& z, const Array<T,1>& x, const Array<T,1>& y) {
  return std::make_pair(g*y, g*x);
}

/**
 * Matrix Frobenius product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $\langle A, A \rangle_\mathrm{F} = \mathrm{Tr}(A^\top A) =
 * \sum_{ij} A_{ij}^2$ as a scalar.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> frobenius(const Array<T,2>& A) {
  return frobenius(A, A);
}

/**
 * Gradient of frobenius().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = \langle A, A \rangle_\mathrm{F} =
 * \mathrm{Tr}(A^\top A) = \sum_{ij} A_{ij}^2$ as a scalar.
 * @param A Matrix $A$.
 * 
 * @return Gradients with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> frobenius_grad(const Array<T,0>& g, const Array<T,0>& z,
    const Array<T,2>& A) {
  return T(2.0)*g*A;
}

/**
 * Matrix-matrix Frobenius product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $\langle A, B \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) =
 * \sum_{ij} A_{ij} B_{ij}$ as a scalar.
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
 * @param z Result $z = \langle A, B \rangle_\mathrm{F} =
 * \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$ as a scalar.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> frobenius_grad(const Array<T,0>& g,
    const Array<T,0>& z, const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(g*B, g*A);
}

/**
 * Matrix-vector inner product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Result $y = A^\top x$.
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
 * @param y Result $y = A^\top x$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p A and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> inner_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& A, const Array<T,1>& x) {
  return std::make_pair(outer(x, g), A*g);
}

/**
 * Matrix-vector inner product and addition.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * @param A Matrix $A$.
 * @param y Vector $y$.
 * 
 * @return Result $z = x + A^\top y$.
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
 * @param z Result $z = x + A^\top y$.
 * @param x Vector $x$.
 * @param A Matrix $A$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p x, @p A and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,1>,Array<T,2>,Array<T,1>> inner_grad(const Array<T,1>& g,
    const Array<T,1>& z, const Array<T,1>& x, const Array<T,2>& A,
    const Array<T,1>& y) {
  return std::make_tuple(g, outer(y, g), A*g);
}

/**
 * Matrix inner product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $B = A^\top A$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inner(const Array<T,2>& A) {
  return inner(A, A);
}

/**
 * Gradient of inner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = A^\top A$.
 * @param A Matrix $A$.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inner_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& A) {
  return A*(g + transpose(g));
}

/**
 * Matrix-matrix inner product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = A^\top B$.
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
 * @param g Gradient with respect to result.
 * @param C Result $C = A^\top B$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> inner_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(outer(B, g), A*g);
}

/**
 * Matrix-matrix inner product and addition.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Result $D = A + B^\top C$.
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
 * @param g Gradient with respect to result.
 * @param D Result $D = A + B^\top C$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p A, @p B and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,2>,Array<T,2>,Array<T,2>> inner_grad(const Array<T,2>& g,
    const Array<T,2>& D, const Array<T,2>& A, const Array<T,2>& B,
    const Array<T,2>& C) {
  return std::make_tuple(g, outer(C, g), B*g);
}

/**
 * Inverse of a square matrix.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Square matrix $A$.
 * 
 * @return Result $B = A^{-1}$.
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
 * @param g Gradient with respect to result.
 * @param B Result $B = A^{-1}$.
 * @param A Square matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> inv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& A);

/**
 * Logarithm of the determinant of a symmetric positive definite matrix via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive, so its logarithm is defined.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Result $\log(\det S) = \log(\det LL^\top) = 2 \log(\det L)$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,0> lcholdet(const Array<T,2>& L);

/**
 * Gradient of `lcholdet()`.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param d Result $d = \log(\det S)$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> lcholdet_grad(const Array<T,0>& g, const Array<T,0>& d,
    const Array<T,2>& L);

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $\log |\det A|$.
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
 * @param d Result $d = \log |\det A|$.
 * @param A Matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> ldet_grad(const Array<T,0>& g, const Array<T,0>& d,
    const Array<T,2>& A) {
  return g*transpose(inv(A));
}

/**
 * Vector outer product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * 
 * @return Result $B = xx^\top$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,1>& x) {
  return outer(x, x);
}

/**
 * Gradient of outer().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = xx^\top$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> outer_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,1>& x) {
  return (g + transpose(g))*x;
}

/**
 * Vector-vector outer product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $C = xy^\top$.
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
 * @param g Gradient with respect to result.
 * @param C Result $C = xy^\top$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p x and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,1>,Array<T,1>> outer_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,1>& x, const Array<T,1>& y) {
  return std::make_pair(g*y, inner(g, x));
}

/**
 * Vector-vector outer product and addition.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $C = A + xy^\top$.
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
 * @param g Gradient with respect to result.
 * @param C Result $C = A + xy^\top$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p A, @p x and @p y.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,2>,Array<T,1>,Array<T,1>> outer_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,1>& x,
    const Array<T,1>& y) {
  return std::make_tuple(g, g*y, inner(g, x));
}

/**
 * Matrix outer product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $B = AA^\top$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer(const Array<T,2>& A) {
  return outer(A, A);
}

/**
 * Gradient of outer().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = AA^\top$.
 * @param A Matrix $A$.
 * 
 * @return Gradients with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> outer_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& A) {
  return (g + transpose(g))*A;
}

/**
 * Matrix-matrix outer product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = AB^\top$.
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
 * @param g Gradient with respect to result.
 * @param C Result $C = AB^\top$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p A and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> outer_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& B) {
  return std::make_pair(g*B, inner(g, A));
}

/**
 * Matrix-matrix outer product and addition.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Result $D = A + BC^\top$.
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
 * @param g Gradient with respect to result.
 * @param D Result $D = A + BC^\top$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p A, @p B and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::tuple<Array<T,2>,Array<T,2>,Array<T,2>> outer_grad(const Array<T,2>& g,
    const Array<T,2>& D, const Array<T,2>& A, const Array<T,2>& B,
    const Array<T,2>& C) {
  return std::make_tuple(g, g*C, inner(g, B));
}

/**
 * Matrix transpose.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $B = A^\top$.
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
 * @param g Gradient with respect to result.
 * @param B Result $B = A^\top$.
 * @param A Matrix.
 * 
 * @return Gradient with respect to @p A.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> transpose_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& A) {
  return transpose(g);
}

/**
 * Lower-triangular-matrix-vector inner product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Result $y = Lx$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> triinner(const Array<T,2>& L, const Array<T,1>& x);

/**
 * Gradient of triinner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = L^\top x$.
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p L and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> triinner_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& L, const Array<T,1>& x);

/**
 * Lower-triangular-matrix inner product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $C = L^\top L$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triinner(const Array<T,2>& L);

/**
 * Gradient of triinner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = L^\top L$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triinner_grad(const Array<T,2>& g, const Array<T,2>& C,
    const Array<T,2>& L);

/**
 * Lower-triangular-matrix-matrix inner product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = L^\top B$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triinner(const Array<T,2>& L, const Array<T,2>& B);

/**
 * Gradient of triinner().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = L^\top B$.
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p L and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> triinner_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& L, const Array<T,2>& B);

/**
 * Inverse of a triangular matrix.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $L^{-1}$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triinv(const Array<T,2>& L);

/**
 * Gradient of triinv().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = L^{-1}$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triinv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& L);

/**
 * Lower-triangular-matrix-vector product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Result $y = Lx$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> trimul(const Array<T,2>& L, const Array<T,1>& x);

/**
 * Gradient of trimul().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = Lx$.
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Gradients with respect to @p L and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> trimul_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& L, const Array<T,1>& x);

/**
 * Lower-triangular-matrix-matrix product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = LB$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> trimul(const Array<T,2>& L, const Array<T,2>& B);

/**
 * Gradient of trimul().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = LB$.
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Gradients with respect to @p L and @p B.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> trimul_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& L, const Array<T,2>& B);

/**
 * Lower-triangular-matrix outer product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $C = LL^\top$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triouter(const Array<T,2>& L);

/**
 * Gradient of triouter().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = LL^\top$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triouter_grad(const Array<T,2>& g, const Array<T,2>& C,
    const Array<T,2>& L);

/**
 * Matrix-lower-triangular-matrix outer product.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix $A$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $C = AL^\top$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> triouter(const Array<T,2>& A, const Array<T,2>& L);

/**
 * Gradient of triouter().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AL^\top$.
 * @param A Matrix $A$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradients with respect to @p A and @p L.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> triouter_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& L);

/**
 * Lower-triangular-matrix-vector solve.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Solution of $x$ in $Lx = y$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,1> trisolve(const Array<T,2>& L, const Array<T,1>& y);

/**
 * Gradient of trisolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $Lx = y$.
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Gradients with respect to @p L and @p x.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,1>> trisolve_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& L, const Array<T,1>& y);

/**
 * Lower-triangular-matrix-matrix solve.
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Solution of $B$ in $LB = C$.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
Array<T,2> trisolve(const Array<T,2>& L, const Array<T,2>& C);

/**
 * Gradient of trisolve().
 * 
 * @ingroup la
 * 
 * @tparam T Floating point type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = C$.
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Gradients with respect to @p L and @p C.
 */
template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
std::pair<Array<T,2>,Array<T,2>> trisolve_grad(const Array<T,2>& g,
    const Array<T,2>& B, const Array<T,2>& L, const Array<T,2>& C);

}
