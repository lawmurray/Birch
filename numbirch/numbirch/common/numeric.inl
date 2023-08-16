
#pragma once

#include "numbirch/numeric.hpp"
#include "numbirch/array.hpp"
#include "numbirch/transform.hpp"

namespace numbirch {

Array<real,2> mul_grad1(const Array<real,1>& g, const Array<real,2>& A,
    const Array<real,1>& x) {
  return outer(g, x);
}

Array<real,1> mul_grad2(const Array<real,1>& g, const Array<real,2>& A,
    const Array<real,1>& x) {
  return inner(A, g);
}

Array<real,2> mul_grad1(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return outer(g, B);
}

Array<real,2> mul_grad2(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return inner(A, g);
}

Array<real,2> chol_grad(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& S) {
  auto A = phi(triinner(L, g));
  return phi(transpose(triinnersolve(L, transpose(triinnersolve(L, add(A,
      transpose(A)))))));
}

Array<real,2> cholinv(const Array<real,2>& L) {
  return cholsolve(L, real(1));
}

Array<real,2> cholinv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L) {
  return cholsolve_grad1(g, B, L, real(1));
}

Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const real& y) {
  auto gy = cholsolve(L, g);
  auto gS = outer(gy, neg(B));
  auto gL = tri(mul(add(gS, transpose(gS)), L));
  return gL;
}

Array<real,0> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const real& y) {
  return sum(cholsolve(L, g));
}

Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,0>& y) {
  auto gy = cholsolve(L, g);
  auto gS = outer(gy, neg(B));
  auto gL = tri(mul(add(gS, transpose(gS)), L));
  return gL;
}

Array<real,0> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,0>& y) {
  return sum(cholsolve(L, g));
}

Array<real,2> cholsolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  auto gy = cholsolve(L, g);
  auto gS = outer(gy, neg(x));
  auto gL = tri(mul(add(gS, transpose(gS)), L));
  return gL;
}

Array<real,1> cholsolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return cholsolve(L, g);
}

Array<real,2> cholsolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  auto gC = cholsolve(L, g);
  auto gS = outer(gC, neg(B));
  auto gL = tri(mul(add(gS, transpose(gS)), L));
  return gL;
}

Array<real,2> cholsolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return cholsolve(L, g);
}

Array<real,0> dot(const Array<real,1>& x) {
  return dot(x, x);
}

Array<real,1> dot_grad(const Array<real,0>& g, const Array<real,1>& x) {
  return mul(mul(real(2), g), x);
}

Array<real,1> dot_grad1(const Array<real,0>& g, const Array<real,1>& x,
    const Array<real,1>& y) {
  return mul(g, y);
}

Array<real,1> dot_grad2(const Array<real,0>& g, const Array<real,1>& x,
    const Array<real,1>& y) {
  return mul(g, x);
}

Array<real,0> frobenius(const Array<real,2>& A) {
  return frobenius(A, A);
}

Array<real,2> frobenius_grad(const Array<real,0>& g, const Array<real,2>& A) {
  return mul(mul(real(2), g), A);
}

Array<real,2> frobenius_grad1(const Array<real,0>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return mul(g, B);
}

Array<real,2> frobenius_grad2(const Array<real,0>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return mul(g, A);
}

Array<real,2> inner_grad1(const Array<real,1>& g, const Array<real,2>& A,
    const Array<real,1>& x) {
  return outer(x, g);
}

Array<real,1> inner_grad2(const Array<real,1>& g, const Array<real,2>& A,
    const Array<real,1>& x) {
  return mul(A, g);
}

Array<real,2> inner(const Array<real,2>& A) {
  return inner(A, A);
}

Array<real,2> inner_grad(const Array<real,2>& g, const Array<real,2>& A) {
  return mul(A, add(g,  transpose(g)));
}

Array<real,2> inner_grad1(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return outer(B, g);
}

Array<real,2> inner_grad2(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return mul(A, g);
}

Array<real,2> inv_grad(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& A) {
  return neg(outer(inner(B, g), B));
}

Array<real,0> lcholdet(const Array<real,2>& L) {
  return mul(real(2), ltridet(L));
}

Array<real,2> lcholdet_grad(const Array<real,0>& g, const Array<real,2>& L) {
  return ltridet_grad(mul(real(2), g), L);
}

Array<real,2> ldet_grad(const Array<real,0>& g, const Array<real,2>& A) {
  return mul(g, transpose(inv(A)));
}

Array<real,0> ltridet(const Array<real,2>& L) {
  return sum(log(L.diagonal()));
}

Array<real,2> ltridet_grad(const Array<real,0>& g, const Array<real,2>& L) {
  return diagonal(div(g, L.diagonal()));
}

Array<real,2> outer(const Array<real,1>& x) {
  return outer(x, x);
}

Array<real,1> outer_grad(const Array<real,2>& g, const Array<real,1>& x) {
  return mul(add(g, transpose(g)), x);
}

Array<real,1> outer_grad1(const Array<real,2>& g, const Array<real,1>& x,
    const Array<real,1>& y) {
  return mul(g, y);
}

Array<real,1> outer_grad2(const Array<real,2>& g, const Array<real,1>& x,
    const Array<real,1>& y) {
  return inner(g, x);
}

Array<real,2> outer(const Array<real,2>& A) {
  return outer(A, A);
}

Array<real,2> outer_grad(const Array<real,2>& g, const Array<real,2>& A) {
  return mul(add(g, transpose(g)), A);
}

Array<real,2> outer_grad1(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return mul(g, B);
}

Array<real,2> outer_grad2(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& B) {
  return inner(g, A);
}

Array<real,2> phi_grad(const Array<real,2>& g, const Array<real,2>& A) {
  return phi(g);
}

template<arithmetic T>
NUMBIRCH_KEEP Array<real,2> transpose_grad(const Array<real,2>& g, const Array<T,2>& A) {
  return transpose(g);
}

Array<real,2> tri_grad(const Array<real,2>& g, const Array<real,2>& A) {
  return tri(g);
}

Array<real,2> triinner_grad1(const Array<real,1>& g, const Array<real,2>& L,
    const Array<real,1>& x) {
  return tri(outer(x, g));
}

Array<real,1> triinner_grad2(const Array<real,1>& g, const Array<real,2>& L,
    const Array<real,1>& x) {
  return trimul(L, g);
}

Array<real,2> triinner(const Array<real,2>& L) {
  return triinner(L, L);
}

Array<real,2> triinner_grad(const Array<real,2>& g, const Array<real,2>& L) {
  return tri(trimul(L, add(g, transpose(g))));
}

Array<real,2> triinner_grad1(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& B) {
  return tri(outer(B, g));
}

Array<real,2> triinner_grad2(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& B) {
  return trimul(L, g);
}

Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const real& y) {
  return tri(outer(neg(B), trisolve(L, g)));
}

Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,0>& y) {
  return tri(outer(neg(B), trisolve(L, g)));
}

Array<real,0> triinnersolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const real& y) {
  return sum(trisolve(L, g));
}

Array<real,0> triinnersolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,0>& y) {
  return sum(trisolve(L, g));
}

Array<real,2> triinnersolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return tri(outer(neg(x), trisolve(L, g)));
}

Array<real,1> triinnersolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return trisolve(L, g);
}

Array<real,2> triinnersolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return tri(outer(neg(B), trisolve(L, g)));
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

Array<real,2> trimul_grad1(const Array<real,1>& g, const Array<real,2>& L,
    const Array<real,1>& x) {
  return tri(outer(g, x));
}

Array<real,1> trimul_grad2(const Array<real,1>& g, const Array<real,2>& L,
    const Array<real,1>& x) {
  return triinner(L, g);
}

Array<real,2> trimul_grad1(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& B) {
  return tri(outer(g, B));
}

Array<real,2> trimul_grad2(const Array<real,2>& g, const Array<real,2>& L,
    const Array<real,2>& B) {
  return triinner(L, g);
}

Array<real,2> triouter(const Array<real,2>& L) {
  return triouter(L, L);
}

Array<real,2> triouter_grad(const Array<real,2>& g, const Array<real,2>& L) {
  return tri(mul(add(g, transpose(g)), L));
}

Array<real,2> triouter_grad1(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& L) {
  return mul(g, L);
}

Array<real,2> triouter_grad2(const Array<real,2>& g, const Array<real,2>& A,
    const Array<real,2>& L) {
  return tri(inner(g, A));
}

Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const real& y) {
  return tri(outer(triinnersolve(L, g), neg(B)));
}

Array<real,0> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const real& y) {
  return sum(triinnersolve(L, g));
}

Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,0>& y) {
  return tri(outer(triinnersolve(L, g), neg(B)));
}

Array<real,0> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,0>& y) {
  return sum(triinnersolve(L, g));
}

Array<real,2> trisolve_grad1(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return tri(outer(triinnersolve(L, g), neg(x)));
}

Array<real,1> trisolve_grad2(const Array<real,1>& g, const Array<real,1>& x,
    const Array<real,2>& L, const Array<real,1>& y) {
  return triinnersolve(L, g);
}

Array<real,2> trisolve_grad1(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return tri(outer(triinnersolve(L, g), neg(B)));
}

Array<real,2> trisolve_grad2(const Array<real,2>& g, const Array<real,2>& B,
    const Array<real,2>& L, const Array<real,2>& C) {
  return triinnersolve(L, g);
}

}
