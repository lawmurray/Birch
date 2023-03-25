
#pragma once

#include "numbirch/numeric.hpp"
#include "numbirch/array.hpp"
#include "numbirch/transform.hpp"

namespace numbirch {

Array<real,2> mul_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x) {
  return outer(g, x);
}

Array<real,1> mul_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x) {
  return inner(A, g);
}

Array<real,2> mul_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B) {
  return outer(g, B);
}

Array<real,2> mul_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B) {
  return inner(A, g);
}

Array<real,2> chol_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& S) {
  auto A = phi(triinner(L, g));
  return phi(transpose(triinnersolve(L, transpose(triinnersolve(L, A +
      transpose(A))))));
}

Array<real,2> cholinv(const Array<real,2>& L) {
  return cholsolve(L, real(1));
}

Array<real,2> cholinv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L) {
  return cholsolve_grad1(g, B, L, real(1));
}

template<class U, class>
Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y) {
  auto gy = cholsolve(L, g);
  auto gS = outer(gy, -B);
  auto gL = tri((gS + transpose(gS))*L);
  return gL;
}

template<class U, class>
Array<real,0> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y) {
  return sum(cholsolve(L, g));
}

Array<real,2> cholsolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  auto gy = cholsolve(L, g);
  auto gS = outer(gy, -x);
  auto gL = tri((gS + transpose(gS))*L);
  return gL;
}

Array<real,1> cholsolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return cholsolve(L, g);
}

Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  auto gC = cholsolve(L, g);
  auto gS = outer(gC, -B);
  auto gL = tri((gS + transpose(gS))*L);
  return gL;
}

Array<real,2> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return cholsolve(L, g);
}

Array<real,0> dot(const Array<real,1>& x) {
  return dot(x, x);
}

Array<real,1> dot_grad(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,1>& x) {
  return real(2)*g*x;
}

Array<real,1> dot_grad1(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,1>& x, const Array<real,1>& y) {
  return g*y;
}

Array<real,1> dot_grad2(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,1>& x, const Array<real,1>& y) {
  return g*x;
}

Array<real,0> frobenius(const Array<real,2>& A) {
  return frobenius(A, A);
}

Array<real,2> frobenius_grad(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,2>& A) {
  return real(2)*g*A;
}

Array<real,2> frobenius_grad1(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,2>& A, const Array<real,2>& B) {
  return g*B;
}

Array<real,2> frobenius_grad2(const Array<real,0>& g, const Array<real,0>& z,
    const Array<real,2>& A, const Array<real,2>& B) {
  return g*A;
}

Array<real,2> inner_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x) {
  return outer(x, g);
}

Array<real,1> inner_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& A, const Array<real,1>& x) {
  return A*g;
}

Array<real,2> inner(const Array<real,2>& A) {
  return inner(A, A);
}

Array<real,2> inner_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A) {
  return A*(g + transpose(g));
}

Array<real,2> inner_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B) {
  return outer(B, g);
}

Array<real,2> inner_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B) {
  return A*g;
}

Array<real,2> inv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A) {
  return -outer(inner(B, g), B);
}

Array<real,0> lcholdet(const Array<real,2>& L) {
  return real(2)*ltridet(L);
}

Array<real,2> lcholdet_grad(const Array<real,0>& g, const Array<real,0>& d,
    const Array<real,2>& L) {
  return ltridet_grad(real(2)*g, d, L);
}

Array<real,2> ldet_grad(const Array<real,0>& g, const Array<real,0>& d,
    const Array<real,2>& A) {
  return g*transpose(inv(A));
}

Array<real,0> ltridet(const Array<real,2>& L) {
  return sum(log(L.diagonal()));
}

Array<real,2> ltridet_grad(const Array<real,0>& g, const Array<real,0>& d,
    const Array<real,2>& L) {
  return diagonal(g/L.diagonal());
}

Array<real,2> outer(const Array<real,1>& x) {
  return outer(x, x);
}

Array<real,1> outer_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,1>& x) {
  return (g + transpose(g))*x;
}

Array<real,1> outer_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,1>& x, const Array<real,1>& y) {
  return g*y;
}

Array<real,1> outer_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,1>& x, const Array<real,1>& y) {
  return inner(g, x);
}

Array<real,2> outer(const Array<real,2>& A) {
  return outer(A, A);
}

Array<real,2> outer_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A) {
  return (g + transpose(g))*A;
}

Array<real,2> outer_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B) {
  return g*B;
}

Array<real,2> outer_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& B) {
  return inner(g, A);
}

Array<real,2> phi_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& A) {
  return phi(g);
}

Array<real,2> transpose_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A) {
  return transpose(g);
}

Array<real,2> tri_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& A) {
  return tri(g);
}

Array<real,2> triinner_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x) {
  return tri(outer(x, g));
}

Array<real,1> triinner_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x) {
  return trimul(L, g);
}

Array<real,2> triinner(const Array<real,2>& L) {
  return triinner(L, L);
}

Array<real,2> triinner_grad(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L) {
  return tri(trimul(L, g + transpose(g)));
}

Array<real,2> triinner_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B) {
  return tri(outer(B, g));
}

Array<real,2> triinner_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B) {
  return trimul(L, g);
}

template<class U, class>
Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y) {
  return tri(outer(-B, trisolve(L, g)));
}

template<class U, class>
Array<real,0> triinnersolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y) {
  return sum(trisolve(L, g));
}

Array<real,2> triinnersolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return tri(outer(-x, trisolve(L, g)));
}

Array<real,1> triinnersolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return trisolve(L, g);
}

Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return tri(outer(-B, trisolve(L, g)));
}

Array<real,2> triinnersolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return trisolve(L, g);
}

Array<real,2> triinv(const Array<real,2>& L) {
  return trisolve(L, real(1));
}

Array<real,2> triinv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L) {
  return trisolve_grad1(g, B, L, real(1));
}

Array<real,2> trimul_grad1(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x) {
  return tri(outer(g, x));
}

Array<real,1> trimul_grad2(const Array<real,1>& g, const Array<real,1>& y,
    const Array<real,2>& L, const Array<real,1>& x) {
  return triinner(L, g);
}

Array<real,2> trimul_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B) {
  return tri(outer(g, B));
}

Array<real,2> trimul_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L, const Array<real,2>& B) {
  return triinner(L, g);
}

Array<real,2> triouter(const Array<real,2>& L) {
  return triouter(L, L);
}

Array<real,2> triouter_grad(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& L) {
  return tri((g + transpose(g))*L);
}

Array<real,2> triouter_grad1(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& L) {
  return g*L;
}

Array<real,2> triouter_grad2(const Array<real,2>& g, const Array<real,2>& C,
    const Array<real,2>& A, const Array<real,2>& L) {
  return tri(inner(g, A));
}

template<class U, class>
Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y) {
  return tri(outer(triinnersolve(L, g), -B));
}

template<class U, class>
Array<real,0> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const U& y) {
  return sum(triinnersolve(L, g));
}

Array<real,2> trisolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return tri(outer(triinnersolve(L, g), -x));
}

Array<real,1> trisolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return triinnersolve(L, g);
}

Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return tri(outer(triinnersolve(L, g), -B));
}

Array<real,2> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return triinnersolve(L, g);
}

}
