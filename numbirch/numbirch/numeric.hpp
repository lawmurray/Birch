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
 * Unary plus.
 * 
 * @ingroup linalg
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @see pos()
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator+(const T& x) {
  return pos(x);
}

/**
 * Negation.
 * 
 * @ingroup linalg
 * 
 * @tparam T Numeric type.
 * 
 * @param x Argument.
 * 
 * @return Result.
 * 
 * @see neg()
 */
template<class T, class = std::enable_if_t<is_numeric_v<T>,int>>
T operator-(const T& x) {
  return neg(x);
}

/**
 * Element-wise addition.
 * 
 * @ingroup linalg
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @see add()
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
implicit_t<T,U> operator+(const T& x, const U& y) {
  /* optimizations for addition of scalar zero */
  if constexpr (is_arithmetic_v<T>) {
    if (value(x) == 0) {
      return y;
    }
  } else if constexpr (is_arithmetic_v<U>) {
    if (value(y) == 0) {
      return x;
    }
  }
  return add(x, y);
}

/**
 * Element-wise subtraction.
 * 
 * @ingroup linalg
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @see sub()
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
implicit_t<T,U> operator-(const T& x, const U& y) {
  /* optimization for subtraction of scalar zero */
  if constexpr (is_arithmetic_v<U>) {
    if (value(y) == 0) {
      return x;
    }
  }
  return sub(x, y);
}

/**
 * Scalar multiplication.
 * 
 * @ingroup linalg
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @note operator*() supports only multiplication by a scalar on the left or
 * right; for element-wise multiplication, see hadamard().
 * 
 * @see hadamard()
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
implicit_t<T,U> operator*(const T& x, const U& y) {
  /* optimizations for multiplication of scalar one */
  if constexpr (is_arithmetic_v<T>) {
    if (value(x) == 1) {
      return y;
    }
  } else if constexpr (is_arithmetic_v<U>) {
    if (value(y) == 1) {
      return x;
    }
  }
  return hadamard(x, y);
}

/**
 * Gradient of operator*().
 * 
 * @ingroup transform_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
real_t<T> mul_grad1(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  /* optimization for multiplication of scalar one */
  if constexpr (is_arithmetic_v<U>) {
    if (value(y) == 1) {
      return g;
    }
  }
  return hadamard_grad1(g, z, x, y);
}

/**
 * Gradient of operator*().
 * 
 * @ingroup transform_grad
 * 
 * @tparam T Numeric type.
 * @tparam U Numeric type.
 * 
 * @param g Gradient with respect to result.
 * @param z Result.
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Gradient with respect to @p y.
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U> && (is_scalar_v<T> || is_scalar_v<U>),int>>
real_t<U> mul_grad2(const real_t<T,U>& g, const implicit_t<T,U>& z, const T& x,
    const U& y) {
  /* optimization for multiplication of scalar one */
  if constexpr (is_arithmetic_v<T>) {
    if (value(x) == 1) {
      return g;
    }
  }
  return hadamard_grad2(g, z, x, y);
}

/**
 * Matrix-vector multiplication.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Result $y = Ax$.
 */
Array<real,1> operator*(const Array<real,2>& A, const Array<real,1>& x);

/**
 * Gradient of operator*().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = Ax$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> mul_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x);

/**
 * Gradient of operator*().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = Ax$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> mul_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x);

/**
 * Matrix-matrix multiplication.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = AB$.
 */
Array<real,2> operator*(const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of operator*().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AB$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> mul_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of operator*().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AB$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p B.
 */
Array<real,2> mul_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Element-wise division.
 * 
 * @ingroup linalg
 * 
 * @tparam T Numeric type.
 * @tparam U Scalar type.
 * 
 * @param x Argument.
 * @param y Argument.
 * 
 * @return Result.
 * 
 * @see div()
 */
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
implicit_t<T,U> operator/(const T& x, const U& y) {
  /* optimization for division of scalar one */
  if constexpr (is_arithmetic_v<U>) {
    if (value(y) == 1) {
      return x;
    }
  }
  return div(x, y);
}

/**
 * Cholesky factorization of a symmetric positive definite matrix.
 * 
 * @ingroup linalg
 * 
 * @param S Symmetric positive definite matrix $S$.
 * 
 * @return Lower-triangular Cholesky factor $L$ such that $S = LL^\top$. If
 * the factorization fails, then $L$ is filled with NaN.
 */
Array<real,2> chol(const Array<real,2>& S);

/**
 * Gradient of chol().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param L Lower-triangular Cholesky factor $L$ such that $S = LL^\top$.
 * @param S Symmetric positive definite matrix $S$.
 * 
 * @return Gradient with respect to @p S.
 */
Array<real,2> chol_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& S);

/**
 * Inverse of a symmetric positive definite matrix via the Cholesky
 * factorization.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Result $S^{-1} = (LL^\top)^{-1}$.
 */
Array<real,2> cholinv(const Array<real,2>& L);

/**
 * Gradient of cholinv().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = S^{-1} = (LL^\top)^{-1}$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> cholinv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L);

/**
 * Matrix-scalar solve via the Cholesky factorization.
 * 
 * @ingroup linalg
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Scalar $y$.
 * 
 * @return Solution of $B$ in $SB = LL^\top B = Iy$.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,2> cholsolve(const Array<real,2>& L, const U& y);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup linalg_grad
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $SB = LL^\top B = Iy$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Scalar $y$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup linalg_grad
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $SB = LL^\top B = Iy$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Scalar $y$.
 * 
 * @return Gradient with respect to  @p y.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,0> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y);

/**
 * Matrix-vector solve via the Cholesky factorization.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Vector $y$.
 * 
 * @return Solution of $x$ in $Sx = LL^\top x = y$.
 */
Array<real,1> cholsolve(const Array<real,2>& L, const Array<real,1>& y);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $Sx = LL^\top x = y$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> cholsolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $Sx = LL^\top x = y$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to  @p y.
 */
Array<real,1> cholsolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y);

/**
 * Matrix-matrix solve via the Cholesky factorization.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param C Matrix $C$.
 * 
 * @return Solution of $B$ in $SB = LL^\top B = C$.
 */
Array<real,2> cholsolve(const Array<real,2>& L, const Array<real,2>& C);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $SB = LL^\top B = C$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param C Matrix $C$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C);

/**
 * Gradient of cholsolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $SB = LL^\top B = C$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * @param C Matrix $C$.
 * 
 * @return Gradient with respect to @p C.
 */
Array<real,2> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C);

/**
 * Vector dot product.
 * 
 * @ingroup linalg
 * 
 * @param x Vector $x$.
 * 
 * @return Result $x^\top x$ as a scalar; zero for empty $x$.
 */
Array<real,0> dot(const Array<real,1>& x);

/**
 * Gradient of dot().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = x^\top x$ as a scalar.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> dot_grad(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,1>& x);

/**
 * Vector-vector dot product.
 * 
 * @ingroup linalg
 * 
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $x^\top y$ as a scalar; zero for empty $x$ and $y$.
 */
Array<real,0> dot(const Array<real,1>& x, const Array<real,1>& y);

/**
 * Gradient of dot().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = x^\top y$ as a scalar.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> dot_grad1(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,1>& x, const Array<real,1>& y);

/**
 * Gradient of dot().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = x^\top y$ as a scalar.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p y.
 */
Array<real,1> dot_grad2(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,1>& x, const Array<real,1>& y);

/**
 * Matrix Frobenius product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $\langle A, A \rangle_\mathrm{F} = \mathrm{Tr}(A^\top A) =
 * \sum_{ij} A_{ij}^2$ as a scalar; zero for empty $A$.
 */
Array<real,0> frobenius(const Array<real,2>& A);

/**
 * Gradient of frobenius().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = \langle A, A \rangle_\mathrm{F} =
 * \mathrm{Tr}(A^\top A) = \sum_{ij} A_{ij}^2$ as a scalar.
 * @param A Matrix $A$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> frobenius_grad(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,2>& A);

/**
 * Matrix-matrix Frobenius product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $\langle A, B \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) =
 * \sum_{ij} A_{ij} B_{ij}$ as a scalar; zero for empty $A$ and $B$.
 */
Array<real,0> frobenius(const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of frobenius().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = \langle A, B \rangle_\mathrm{F} =
 * \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$ as a scalar.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> frobenius_grad1(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of frobenius().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param z Result $z = \langle A, B \rangle_\mathrm{F} =
 * \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$ as a scalar.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p B.
 */
Array<real,2> frobenius_grad2(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Matrix-vector inner product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Result $y = A^\top x$.
 */
Array<real,1> inner(const Array<real,2>& A, const Array<real,1>& x);

/**
 * Gradient of inner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = A^\top x$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> inner_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x);

/**
 * Gradient of inner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = A^\top x$.
 * @param A Matrix $A$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> inner_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x);

/**
 * Matrix inner product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $B = A^\top A$.
 */
Array<real,2> inner(const Array<real,2>& A);

/**
 * Gradient of inner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = A^\top A$.
 * @param A Matrix $A$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> inner_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A);

/**
 * Matrix-matrix inner product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = A^\top B$.
 */
Array<real,2> inner(const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of inner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = A^\top B$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> inner_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of inner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = A^\top B$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p B.
 */
Array<real,2> inner_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Inverse of a square matrix.
 * 
 * @ingroup linalg
 * 
 * @param A Square matrix $A$.
 * 
 * @return Result $B = A^{-1}$.
 */
Array<real,2> inv(const Array<real,2>& A);

/**
 * Gradient of inv().
 * 
 * @ingroup linalg_grad
 * 
= * @param g Gradient with respect to result.
 * @param B Result $B = A^{-1}$.
 * @param A Square matrix.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> inv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A);

/**
 * Logarithm of the determinant of a symmetric positive definite matrix via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive, so its logarithm is defined.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Result $\log(\det S) = \log(\det LL^\top) = 2 \log(\det L)$; zero
 * for empty $L$.
 */
Array<real,0> lcholdet(const Array<real,2>& L);

/**
 * Gradient of `lcholdet()`.
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param d Result $d = \log(\det S)$.
 * @param L Lower-triangular Cholesky factor $L$ of the symmetric positive
 * definite matrix $S = LL^\top$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> lcholdet_grad(const Array<real,0>& g, const Array<real,0>& d,
    const Array<real,2>& L);

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $\log |\det A|$; zero for empty $A$.
 */
Array<real,0> ldet(const Array<real,2>& A);

/**
 * Gradient of `ldet()`.
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param d Result $d = \log |\det A|$.
 * @param A Matrix.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> ldet_grad(const Array<real,0>& g, const Array<real,0>& d,
    const Array<real,2>& A);

/**
 * Logarithm of the absolute value of the determinant of a lower-triangular
 * matrix.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $\log|\det L|$; zero for empty $L$.
 */
Array<real,0> ltridet(const Array<real,2>& L);

/**
 * Gradient of `ltridet()`.
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param d Result $\log|\det L|$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> ltridet_grad(const Array<real,0>& g, const Array<real,0>& d,
    const Array<real,2>& L);

/**
 * Vector outer product.
 * 
 * @ingroup linalg
 * 
 * @param x Vector $x$.
 * 
 * @return Result $B = xx^\top$.
 */
Array<real,2> outer(const Array<real,1>& x);

/**
 * Gradient of outer().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = xx^\top$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> outer_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,1>& x);

/**
 * Vector-vector outer product.
 * 
 * @ingroup linalg
 * 
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Result $C = xy^\top$.
 */
Array<real,2> outer(const Array<real,1>& x, const Array<real,1>& y);

/**
 * Gradient of outer().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = xy^\top$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> outer_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,1>& x, const Array<real,1>& y);

/**
 * Gradient of outer().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = xy^\top$.
 * @param x Vector $x$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p y.
 */
Array<real,1> outer_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,1>& x, const Array<real,1>& y);

/**
 * Matrix outer product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $B = AA^\top$.
 */
Array<real,2> outer(const Array<real,2>& A);

/**
 * Gradient of outer().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = AA^\top$.
 * @param A Matrix $A$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> outer_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A);

/**
 * Matrix-matrix outer product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = AB^\top$.
 */
Array<real,2> outer(const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of outer().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AB^\top$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> outer_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Gradient of outer().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AB^\top$.
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p A and @p B.
 */
Array<real,2> outer_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B);

/**
 * Extract the lower triangle and half the diagonal of a matrix.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Lower triangle and half the diagonal of $A$. The upper triangle is
 * filled with zeros.
 */
Array<real,2> phi(const Array<real,2>& A);

/**
 * Gradient of phi().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param L Lower triangle and half the diagonal of $A$.
 * @param A Matrix $S$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> phi_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& A);

/**
 * Matrix transpose.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Result $B = A^\top$.
 */
Array<real,2> transpose(const Array<real,2>& A);

/**
 * Gradient of transpose().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = A^\top$.
 * @param A Matrix $A$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> transpose_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A);

/**
 * Scalar transpose.
 * 
 * @ingroup linalg
 * 
 * @param x Scalar $x$.
 * 
 * @return Result $y = x$.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
T transpose(const T& x) {
  return x;
}

/**
 * Gradient of transpose().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = x$.
 * @param x Scalar $x$.
 * 
 * @return Gradient with respect to @p x.
 */
template<class T, class = std::enable_if_t<is_scalar_v<T>,int>>
real_t<T> transpose_grad(const real_t<T>& g, const T& y, const T& x) {
  return g;
}

/**
 * Extract the lower triangle and diagonal of a matrix.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * 
 * @return Lower triangle and diagonal of $A$. The upper triangle is filled
 * with zeros.
 */
Array<real,2> tri(const Array<real,2>& A);

/**
 * Gradient of tri().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param L Lower triangle and diagonal of $A$.
 * @param A Matrix $S$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> tri_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& A);

/**
 * Lower-triangular-matrix-vector inner product.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Result $y = Lx$.
 */
Array<real,1> triinner(const Array<real,2>& L, const Array<real,1>& x);

/**
 * Gradient of triinner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = L^\top x$.
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triinner_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x);

/**
 * Gradient of triinner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = L^\top x$.
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> triinner_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x);

/**
 * Lower-triangular-matrix inner product.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $S = L^\top L$.
 */
Array<real,2> triinner(const Array<real,2>& L);

/**
 * Gradient of triinner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = L^\top L$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triinner_grad(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L);

/**
 * Lower-triangular-matrix-matrix inner product.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = L^\top B$.
 */
Array<real,2> triinner(const Array<real,2>& L, const Array<real,2>& B);

/**
 * Gradient of triinner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = L^\top B$.
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triinner_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B);

/**
 * Gradient of triinner().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = L^\top B$.
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p B.
 */
Array<real,2> triinner_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B);

/**
 * Lower-triangular-matrix-scalar inner solve.
 * 
 * @ingroup linalg
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param y Scalar $y$.
 * 
 * @return Solution of $B$ in $LB = Iy$.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,2> triinnersolve(const Array<real,2>& L, const U& y);

/**
 * Gradient of triinnersolve().
 * 
 * @ingroup linalg_grad
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = Iy$.
 * @param L Lower-triangular matrix $L$.
 * @param y Scalar $y$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y);

/**
 * Gradient of triinnersolve().
 * 
 * @ingroup linalg_grad
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = Iy$.
 * @param L Lower-triangular matrix $L$.
 * @param y Scalar $y$.
 * 
 * @return Gradient with respect to  @p y.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,0> triinnersolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y);

/**
 * Lower-triangular-matrix-vector inner solve.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Solution of $x$ in $y = L^\top x$.
 */
Array<real,1> triinnersolve(const Array<real,2>& L, const Array<real,1>& x);

/**
 * Gradient of triinnersolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $y = L^\top x$.
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triinnersolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y);

/**
 * Gradient of triinnersolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $y = L^\top x$.
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p y.
 */
Array<real,1> triinnersolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y);

/**
 * Lower-triangular-matrix-matrix inner solve.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Solution of $B$ in $C = L^\top B$.
 */
Array<real,2> triinnersolve(const Array<real,2>& L, const Array<real,2>& C);

/**
 * Gradient of triinnersolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $C = L^\top B$.
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C);

/**
 * Gradient of triinnersolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $C = L^\top B$.
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Gradient with respect to @p C.
 */
Array<real,2> triinnersolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C);

/**
 * Inverse of a triangular matrix.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $L^{-1}$.
 */
Array<real,2> triinv(const Array<real,2>& L);

/**
 * Gradient of triinv().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Result $B = L^{-1}$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triinv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L);

/**
 * Lower-triangular-matrix-vector product.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Result $y = Lx$.
 */
Array<real,1> trimul(const Array<real,2>& L, const Array<real,1>& x);

/**
 * Gradient of trimul().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = Lx$.
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> trimul_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x);

/**
 * Gradient of trimul().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param y Result $y = Lx$.
 * @param L Lower-triangular matrix $L$.
 * @param x Vector $x$.
 * 
 * @return Gradient with respect to @p L and @p x.
 */
Array<real,1> trimul_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x);

/**
 * Lower-triangular-matrix-matrix product.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Result $C = LB$.
 */
Array<real,2> trimul(const Array<real,2>& L, const Array<real,2>& B);

/**
 * Gradient of trimul().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = LB$.
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> trimul_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B);

/**
 * Gradient of trimul().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = LB$.
 * @param L Lower-triangular matrix $L$.
 * @param B Matrix $B$.
 * 
 * @return Gradient with respect to @p B.
 */
Array<real,2> trimul_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B);

/**
 * Lower-triangular-matrix outer product.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $S = LL^\top$.
 */
Array<real,2> triouter(const Array<real,2>& L);

/**
 * Gradient of triouter().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = LL^\top$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triouter_grad(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L);

/**
 * Matrix-lower-triangular-matrix outer product.
 * 
 * @ingroup linalg
 * 
 * @param A Matrix $A$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Result $C = AL^\top$.
 */
Array<real,2> triouter(const Array<real,2>& A, const Array<real,2>& L);

/**
 * Gradient of triouter().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AL^\top$.
 * @param A Matrix $A$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p A.
 */
Array<real,2> triouter_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& L);

/**
 * Gradient of triouter().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param C Result $C = AL^\top$.
 * @param A Matrix $A$.
 * @param L Lower-triangular matrix $L$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> triouter_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& L);

/**
 * Lower-triangular-matrix-scalar solve.
 * 
 * @ingroup linalg
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param L Lower-triangular matrix $L$.
 * @param y Scalar $y$.
 * 
 * @return Solution of $B$ in $LB = Iy$.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,2> trisolve(const Array<real,2>& L, const U& y);

/**
 * Gradient of trisolve().
 * 
 * @ingroup linalg_grad
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = Iy$.
 * @param L Lower-triangular matrix $L$.
 * @param y Scalar $y$.
 * 
 * @return Gradient with respect to @p L.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y);

/**
 * Gradient of trisolve().
 * 
 * @ingroup linalg_grad
 * 
 * @tparam U Floating point scalar type.
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = Iy$.
 * @param L Lower-triangular matrix $L$.
 * @param y Scalar $y$.
 * 
 * @return Gradient with respect to  @p y.
 */
template<class U, class = std::enable_if_t<is_real_v<U> &&
    is_scalar_v<U>,int>>
Array<real,0> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y);

/**
 * Lower-triangular-matrix-vector solve.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Solution of $x$ in $Lx = y$.
 */
Array<real,1> trisolve(const Array<real,2>& L, const Array<real,1>& y);

/**
 * Gradient of trisolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $Lx = y$.
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> trisolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y);

/**
 * Gradient of trisolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param x Solution of $x$ in $Lx = y$.
 * @param L Lower-triangular matrix $L$.
 * @param y Vector $y$.
 * 
 * @return Gradient with respect to @p x.
 */
Array<real,1> trisolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y);

/**
 * Lower-triangular-matrix-matrix solve.
 * 
 * @ingroup linalg
 * 
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Solution of $B$ in $LB = C$.
 */
Array<real,2> trisolve(const Array<real,2>& L, const Array<real,2>& C);

/**
 * Gradient of trisolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = C$.
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Gradient with respect to @p L.
 */
Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C);

/**
 * Gradient of trisolve().
 * 
 * @ingroup linalg_grad
 * 
 * @param g Gradient with respect to result.
 * @param B Solution of $B$ in $LB = C$.
 * @param L Lower-triangular matrix $L$.
 * @param C Matrix $C$.
 * 
 * @return Gradient with respect to @p C.
 */
Array<real,2> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C);

}
