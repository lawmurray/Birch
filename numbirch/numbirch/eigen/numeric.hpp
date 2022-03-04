/**
 * @file
 */
#pragma once

#include "numbirch/numeric.hpp"
#include "numbirch/eigen/eigen.hpp"
#include "numbirch/common/functor.hpp"
#include "numbirch/common/get.hpp"

namespace numbirch {

template<class T, class>
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x) {
  assert(columns(A) == length(x));
  Array<T,1> y(make_shape(rows(A)));
  auto A1 = make_eigen(A);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = A1*x1;
  return y;
}

template<class T, class>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& B) {
  assert(columns(A) == rows(B));
  Array<T,2> C(make_shape(rows(A), columns(B)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = A1*B1;
  return C;
}

template<class T, class>
Array<T,2> chol(const Array<T,2>& S) {
  assert(rows(S) == columns(S));
  Array<T,2> L(shape(S));
  auto S1 = make_eigen(S);
  auto L1 = make_eigen(L);
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    L1 = llt.matrixL();
  } else {
    L.fill(T(0.0/0.0));
  }
  return L;
}

template<class T, class>
Array<T,2> chol_grad(const Array<T,2>& g, const Array<T,2>& L,
    const Array<T,2>& S) {
  assert(rows(g) == columns(g));
  assert(rows(L) == columns(L));
  assert(rows(S) == columns(S));
  assert(rows(g) == rows(L));
  assert(rows(g) == rows(S));
  Array<T,2> gS(shape(S));
  auto g1 = make_eigen(g);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto S1 = make_eigen(S);
  auto gS1 = make_eigen(gS);
  gS1 = (U1*g1).template triangularView<Eigen::Lower>();
  gS1.diagonal() *= 0.5;
  gS1 = U1.solve(U1.solve(gS1).transpose());
  gS1 = (gS1.transpose() + gS1).template triangularView<Eigen::Lower>();
  gS1.diagonal() *= 0.5;
  return gS;
}

template<class T, class>
Array<T,2> cholinv(const Array<T,2>& L) {
  assert(rows(L) == columns(L));
  Array<T,2> B(shape(L));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto I1 = B1.Identity(rows(B), columns(B));
  B1.noalias() = U1.solve(L1.solve(I1));
  return B;
}

template<class T, class>
Array<T,2> cholinv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& L) {
  assert(rows(L) == columns(L));
  Array<T,2> gS(shape(L)), gL(shape(L));
  auto g1 = make_eigen(g);
  auto B1 = make_eigen(B);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto gS1 = make_eigen(gS);
  auto gL1 = make_eigen(gL);
  gS1.noalias() = -B1.transpose()*g1*B1.transpose();
  gL1 = ((gS1 + gS1.transpose())*L1).template triangularView<Eigen::Lower>();
  return gL;
}

template<class T, class>
Array<T,1> cholsolve(const Array<T,2>& L, const Array<T,1>& y) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,1> x(shape(y));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  x1.noalias() = U1.solve(L1.solve(y1));
  return x;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> cholsolve_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& L, const Array<T,1>& y) {
  assert(length(g) == length(y));
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,2> gS(shape(L));
  Array<T,2> gL(shape(L));
  Array<T,1> gy(shape(y));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto gS1 = make_eigen(gS);
  auto gL1 = make_eigen(gL);
  auto gy1 = make_eigen(gy);
  gy1.noalias() = U1.solve(L1.solve(g1));
  gS1.noalias() = -gy1*x1.transpose();
  gL1 = ((gS1 + gS1.transpose())*L1).template triangularView<Eigen::Lower>();
  return std::make_pair(gL, gy);
}

template<class T, class>
Array<T,2> cholsolve(const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> B(shape(C));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  B1.noalias() = U1.solve(L1.solve(C1));
  return B;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> cholsolve_grad(const Array<T,2>& g,
    const Array<T,2>& B, const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> gS(shape(L));
  Array<T,2> gL(shape(L));
  Array<T,2> gC(shape(C));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  auto gS1 = make_eigen(gS);
  auto gL1 = make_eigen(gL);
  auto gC1 = make_eigen(gC);
  gC1.noalias() = U1.solve(L1.solve(g1));
  gS1.noalias() = -gC1*B1.transpose();
  gL1 = ((gS1 + gS1.transpose())*L1).template triangularView<Eigen::Lower>();
  return std::make_pair(gL, gC);
}

template<class T, class>
Array<T,0> dot(const Array<T,1>& x, const Array<T,1>& y) {
  assert(length(x) == length(y));
  Array<T,0> z;
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  z = x1.dot(y1);
  return z;
}

template<class T, class>
Array<T,0> frobenius(const Array<T,2>& x, const Array<T,2>& y) {
  Array<T,0> z;
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  z = (x1.array()*y1.array()).sum();
  return z;
}

template<class T, class>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x) {
  assert(rows(A) == length(x));
  Array<T,1> y(make_shape(columns(A)));
  auto A1 = make_eigen(A);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = A1.transpose()*x1;
  return y;
}

template<class T, class>
Array<T,1> inner(const Array<T,1>& x, const Array<T,2>& A,
    const Array<T,1>& y) {
  assert(rows(A) == length(x));
  assert(columns(A) == length(y));
  Array<T,1> z(make_shape(length(x)));
  auto x1 = make_eigen(x);
  auto A1 = make_eigen(A);
  auto y1 = make_eigen(y);
  auto z1 = make_eigen(z);
  z1.noalias() = x1 + A1.transpose()*y1;
  return z;
}

template<class T, class>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B) {
  assert(rows(A) == rows(B));
  Array<T,2> C(make_shape(columns(A), columns(B)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = A1.transpose()*B1;
  return C;
}

template<class T, class>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B,
    const Array<T,2>& C) {
  assert(rows(A) == columns(B));
  assert(columns(A) == columns(C));
  assert(rows(B) == rows(C));
  Array<T,2> D(make_shape(rows(A), columns(A)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  auto D1 = make_eigen(D);
  D1.noalias() = A1 + B1.transpose()*C1;
  return D;
}

template<class T, class>
Array<T,2> inv(const Array<T,2>& A) {
  assert(rows(A) == columns(A));
  Array<T,2> B(shape(A));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  B1.noalias() = A1.inverse();
  return B;
}

template<class T, class>
Array<T,2> inv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& A) {
  assert(rows(B) == columns(B));
  assert(rows(A) == columns(A));
  Array<T,2> gA(shape(A));
  auto g1 = make_eigen(g);
  auto B1 = make_eigen(B);
  auto gA1 = make_eigen(gA);
  gA1.noalias() = -B1.transpose()*g1*B1.transpose();
  return gA;
}

template<class T, class>
Array<T,0> lcholdet(const Array<T,2>& L) {
  auto L1 = make_eigen(L);

  /* we have $S = LL^\top$; the determinant of $S$ is the square of the
   * determinant of $L$, the determinant of $L$ is the product of the
   * elements along the diagonal, as it is a triangular matrix; adjust for
   * log-determinant */
  return T(2.0)*L1.diagonal().array().log().sum();
}

template<class T, class>
Array<T,2> lcholdet_grad(const Array<T,0>& g, const Array<T,0>& d,
    const Array<T,2>& L) {
  Array<T,2> gL(shape(L));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L);
  auto gL1 = make_eigen(gL);
  gL1 = (T(2.0)*g.value()/L1.diagonal().array()).matrix().asDiagonal();
  assert(gL1.isDiagonal());
  return gL;
}

template<class T, class>
Array<T,0> ldet(const Array<T,2>& A) {
  auto A1 = make_eigen(A);
  return A1.householderQr().logAbsDeterminant();
}

template<class T, class>
Array<T,2> outer(const Array<T,1>& x, const Array<T,1>& y) {
  Array<T,2> A(make_shape(length(x), length(y)));
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto A1 = make_eigen(A);
  A1.noalias() = x1*y1.transpose();
  return A;
}

template<class T, class>
Array<T,2> outer(const Array<T,2>& A, const Array<T,1>& x,
    const Array<T,1>& y) {
  assert(rows(A) == length(x));
  assert(columns(A) == length(y));
  Array<T,2> B(make_shape(rows(A), columns(A)));
  auto A1 = make_eigen(A);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto B1 = make_eigen(B);
  B1.noalias() = A1 + x1*y1.transpose();
  return B;
}

template<class T, class>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B) {
  assert(columns(A) == columns(B));
  Array<T,2> C(make_shape(rows(A), rows(B)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = A1*B1.transpose();
  return C;
}

template<class T, class>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B,
    const Array<T,2>& C) {
  assert(rows(A) == rows(B));
  assert(columns(A) == rows(C));
  assert(columns(B) == columns(C));
  Array<T,2> D(make_shape(rows(A), columns(A)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  auto D1 = make_eigen(D);
  D1.noalias() = A1 + B1*C1.transpose();
  return D;
}

template<class T, class>
Array<T,2> transpose(const Array<T,2>& A) {
  Array<T,2> B(make_shape(columns(A), rows(A)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  B1.noalias() = A1.transpose();
  return B;
}

template<class T, class>
Array<T,1> triinner(const Array<T,2>& L, const Array<T,1>& x) {
  assert(rows(L) == length(x));
  Array<T,1> y(make_shape(columns(L)));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = U1*x1;
  return y;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> triinner_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& L, const Array<T,1>& x) {
  assert(rows(L) == length(x));
  Array<T,2> gL(shape(L));
  Array<T,1> gx(shape(x));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto x1 = make_eigen(x);
  auto gL1 = make_eigen(gL);
  auto gx1 = make_eigen(gx);
  gL1 = (x1*g1.transpose()).template triangularView<Eigen::Lower>();
  gx1.noalias() = L1*g1;
  return std::make_pair(gL, gx);
}

template<class T, class>
Array<T,2> triinner(const Array<T,2>& L) {
  return triinner(L, L);
}

template<class T, class>
Array<T,2> triinner_grad(const Array<T,2>& g, const Array<T,2>& C,
    const Array<T,2>& L) {
  Array<T,2> gL(shape(L));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto gL1 = make_eigen(gL);
  gL1 = (L1*(g1 + g1.transpose())).template triangularView<Eigen::Lower>();
  return gL;
}

template<class T, class>
Array<T,2> triinner(const Array<T,2>& L, const Array<T,2>& B) {
  assert(rows(L) == rows(B));
  Array<T,2> C(make_shape(columns(L), columns(B)));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = U1*B1;
  return C;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> triinner_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& L, const Array<T,2>& B) {
  Array<T,2> gL(shape(L));
  Array<T,2> gB(shape(B));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  auto gL1 = make_eigen(gL);
  auto gB1 = make_eigen(gB);
  gL1 = (B1*g1.transpose()).template triangularView<Eigen::Lower>();
  gB1.noalias() = L1*g1;
  return std::make_pair(gL, gB);
}

template<class T, class>
Array<T,2> triinv(const Array<T,2>& L) {
  assert(rows(L) == columns(L));
  Array<T,2> B(shape(L));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  B1.noalias() = L1.solve(B1.Identity(rows(B), columns(B)));
  return B;
}

template<class T, class>
Array<T,2> triinv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& L) {
  assert(rows(B) == columns(B));
  assert(rows(L) == columns(L));
  Array<T,2> gL(shape(L));
  auto g1 = make_eigen(g);
  auto B1 = make_eigen(B).transpose().template triangularView<Eigen::Upper>();
  auto gL1 = make_eigen(gL);
  gL1 = (-(B1*g1*B1)).template triangularView<
      Eigen::Lower>();
  return gL;
}

template<class T, class>
Array<T,1> trimul(const Array<T,2>& L, const Array<T,1>& x) {
  assert(columns(L) == length(x));
  Array<T,1> y(make_shape(rows(L)));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = L1*x1;
  return y;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> trimul_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& L, const Array<T,1>& x) {
  assert(columns(L) == length(x));
  Array<T,2> gL(shape(L));
  Array<T,1> gx(shape(x));
  auto g1 = make_eigen(g);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto gL1 = make_eigen(gL);
  auto gx1 = make_eigen(gx);
  gL1 = (g1*x1.transpose()).template triangularView<Eigen::Lower>();
  gx1.noalias() = U1*g1;
  return std::make_pair(gL, gx);
}

template<class T, class>
Array<T,2> trimul(const Array<T,2>& L, const Array<T,2>& B) {
  assert(columns(L) == rows(B));
  Array<T,2> C(make_shape(rows(L), columns(B)));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = L1*B1;
  return C;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> trimul_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& L, const Array<T,2>& B) {
  assert(columns(L) == rows(B));
  Array<T,2> gL(shape(L));
  Array<T,2> gB(shape(B));
  auto g1 = make_eigen(g);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto gL1 = make_eigen(gL);
  auto gB1 = make_eigen(gB);
  gL1 = (g1*B1.transpose()).template triangularView<Eigen::Lower>();
  gB1.noalias() = U1*g1;
  return std::make_pair(gL, gB);
}

template<class T, class>
Array<T,2> triouter(const Array<T,2>& L) {
  return triouter(L, L);
}

template<class T, class>
Array<T,2> triouter_grad(const Array<T,2>& g, const Array<T,2>& C,
    const Array<T,2>& L) {
  Array<T,2> gL(shape(L));
  auto g1 = make_eigen(g);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto gL1 = make_eigen(gL);
  gL1 = ((g1 + g1.transpose())*L1).template triangularView<Eigen::Lower>();
  return gL;
}

template<class T, class>
Array<T,2> triouter(const Array<T,2>& A, const Array<T,2>& L) {
  assert(columns(A) == columns(L));
  Array<T,2> C(make_shape(rows(A), rows(L)));
  auto A1 = make_eigen(A);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto C1 = make_eigen(C);
  C1.noalias() = A1*U1;
  return C;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> triouter_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& L) {
  Array<T,2> gA(shape(A));
  Array<T,2> gL(shape(L));
  auto g1 = make_eigen(g);
  auto A1 = make_eigen(A);
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto gA1 = make_eigen(gA);
  auto gL1 = make_eigen(gL);
  gA1.noalias() = g1*L1;
  gL1 = (g1.transpose()*A1).template triangularView<Eigen::Lower>();
  return std::make_pair(gA, gL);
}

template<class T, class>
Array<T,1> trisolve(const Array<T,2>& L, const Array<T,1>& y) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,1> x(shape(y));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  x1.noalias() = L1.solve(y1);
  return x;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> trisolve_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& L, const Array<T,1>& y) {
  assert(length(g) == length(y));
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,2> gL(shape(L));
  Array<T,1> gy(shape(y));
  auto g1 = make_eigen(g);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto gL1 = make_eigen(gL);
  auto gy1 = make_eigen(gy);
  gy1.noalias() = U1.solve(g1);
  gL1 = (-gy1*x1.transpose()).template triangularView<Eigen::Lower>();
  return std::make_pair(gL, gy);
}

template<class T, class>
Array<T,2> trisolve(const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> B(shape(C));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  B1.noalias() = L1.solve(C1);
  return B;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> trisolve_grad(const Array<T,2>& g,
    const Array<T,2>& B, const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(g) == rows(L));
  assert(columns(g) == columns(C));
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> gL(shape(L));
  Array<T,2> gC(shape(C));
  auto g1 = make_eigen(g);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  auto gL1 = make_eigen(gL);
  auto gC1 = make_eigen(gC);
  gC1.noalias() = U1.solve(g1);
  gL1 = (-gC1*B1.transpose()).template triangularView<Eigen::Lower>();
  return std::make_pair(gL, gC);
}

}
