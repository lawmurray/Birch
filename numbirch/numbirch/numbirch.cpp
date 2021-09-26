/**
 * @file
 * 
 * Implementation of numeric functions by instantiating the generic
 * implementations of the chosen backend for double and single precision
 * overloads.
 */
#include "numbirch/numbirch.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/numbirch.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numbirch.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numbirch.hpp"
#endif

namespace numbirch {

void neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB) {
  neg<double>(m, n, A, ldA, B, ldB);
}

void neg(const int m, const int n, const float* A, const int ldA,
    float* B, const int ldB) {
  neg<float>(m, n, A, ldA, B, ldB);
}

void rectify(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB) {
  rectify<double>(m, n, A, ldA, B, ldB);
}

void rectify(const int m, const int n, const float* A, const int ldA,
    float* B, const int ldB) {
  rectify<float>(m, n, A, ldA, B, ldB);
}

void add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  add<double>(m, n, A, ldA, B, ldB, C, ldC);
}

void add(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC) {
  add<float>(m, n, A, ldA, B, ldB, C, ldC);
}

void sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  sub<double>(m, n, A, ldA, B, ldB, C, ldC);
}

void sub(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC) {
  sub<float>(m, n, A, ldA, B, ldB, C, ldC);
}

void combine(const int m, const int n, const double a, const double* A,
    const int ldA, const double b, const double* B, const int ldB,
    const double c, const double* C, const int ldC, const double d,
    const double* D, const int ldD, double* E, const int ldE) {
  combine<double>(m, n, a, A, ldA, b, B, ldB, c, C, ldC, d, D, ldD, E, ldE);
}

void combine(const int m, const int n, const float a, const float* A,
    const int ldA, const float b, const float* B, const int ldB,
    const float c, const float* C, const int ldC, const float d,
    const float* D, const int ldD, float* E, const int ldE) {
  combine<float>(m, n, a, A, ldA, b, B, ldB, c, C, ldC, d, D, ldD, E, ldE);
}

void hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  hadamard<double>(m, n, A, ldA, B, ldB, C, ldC);
}

void hadamard(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC) {
  hadamard<float>(m, n, A, ldA, B, ldB, C, ldC);
}

void div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC) {
  div<double>(m, n, A, ldA, b, C, ldC);
}

void div(const int m, const int n, const float* A, const int ldA,
    const float b, float* C, const int ldC) {
  div<float>(m, n, A, ldA, b, C, ldC);
}

void mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC) {
  mul<double>(m, n, a, B, ldB, C, ldC);
}

void mul(const int m, const int n, const float a, const float* B,
    const int ldB, float* C, const int ldC) {
  mul<float>(m, n, a, B, ldB, C, ldC);
}

void mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  mul<double>(m, n, A, ldA, x, incx, y, incy);
}

void mul(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy) {
  mul<float>(m, n, A, ldA, x, incx, y, incy);
}

void mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  mul<double>(m, n, k, A, ldA, B, ldB, C, ldC);
}

void mul(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC) {
  mul<float>(m, n, k, A, ldA, B, ldB, C, ldC);
}

void cholmul(const int n, const double* S, const int ldS,
    const double* x, const int incx, double* y, const int incy) {
  cholmul<double>(n, S, ldS, x, incx, y, incy);
}

void cholmul(const int n, const float* S, const int ldS,
    const float* x, const int incx, float* y, const int incy) {
  cholmul<float>(n, S, ldS, x, incx, y, incy);
}

void cholmul(const int m, const int n, const double* S,
    const int ldS, const double* B, const int ldB, double* C, const int ldC) {
  cholmul<double>(m, n, S, ldS, B, ldB, C, ldC);
}

void cholmul(const int m, const int n, const float* S,
    const int ldS, const float* B, const int ldB, float* C, const int ldC) {
  cholmul<float>(m, n, S, ldS, B, ldB, C, ldC);
}

double sum(const int m, const int n, const double* A, const int ldA) {
  return sum<double>(m, n, A, ldA);
}

float sum(const int m, const int n, const float* A, const int ldA) {
  return sum<float>(m, n, A, ldA);
}

double dot(const int n, const double* x, const int incx, const double* y,
    const int incy) {
  return dot<double>(n, x, incx, y, incy);
}

float dot(const int n, const float* x, const int incx, const float* y,
    const int incy) {
  return dot<float>(n, x, incx, y, incy);
}

double frobenius(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB) {
  return frobenius<double>(m, n, A, ldA, B, ldB);
}

float frobenius(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB) {
  return frobenius<float>(m, n, A, ldA, B, ldB);
}

void inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  inner<double>(m, n, A, ldA, x, incx, y, incy);
}

void inner(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy) {
  inner<float>(m, n, A, ldA, x, incx, y, incy);
}

void inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  inner<double>(m, n, k, A, ldA, B, ldB, C, ldC);
}

void inner(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC) {
  inner<float>(m, n, k, A, ldA, B, ldB, C, ldC);
}

void outer(const int m, const int n, const double* x, const int incx,
    const double* y, const int incy, double* A, const int ldA) {
  outer<double>(m, n, x, incx, y, incy, A, ldA);
}

void outer(const int m, const int n, const float* x,
    const int incx, const float* y, const int incy, float* A,
    const int ldA) {
  outer<float>(m, n, x, incx, y, incy, A, ldA);
}

void outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  outer<double>(m, n, k, A, ldA, B, ldB, C, ldC);
}

void outer(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC) {
  outer<float>(m, n, k, A, ldA, B, ldB, C, ldC);
}

void cholouter(const int m, const int n, const double* A,
    const int ldA, const double* S, const int ldS, double* C, const int ldC) {
  cholouter<double>(m, n, A, ldA, S, ldS, C, ldC);
}

void cholouter(const int m, const int n, const float* A,
    const int ldA, const float* S, const int ldS, float* C, const int ldC) {
  cholouter<float>(m, n, A, ldA, S, ldS, C, ldC);
}

void solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy) {
  solve<double>(n, A, ldA, x, incx, y, incy);
}

void solve(const int n, const float* A, const int ldA, float* x,
    const int incx, const float* y, const int incy) {
  solve<float>(n, A, ldA, x, incx, y, incy);
}

void solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY) {
  solve<double>(m, n, A, ldA, X, ldX, Y, ldY);
}

void solve(const int m, const int n, const float* A, const int ldA,
    float* X, const int ldX, const float* Y, const int ldY) {
  solve<float>(m, n, A, ldA, X, ldX, Y, ldY);
}

void cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy) {
  cholsolve<double>(n, S, ldS, x, incx, y, incy);
}

void cholsolve(const int n, const float* S, const int ldS,
    float* x, const int incx, const float* y, const int incy) {
  cholsolve<float>(n, S, ldS, x, incx, y, incy);
}

void cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY) {
  cholsolve<double>(m, n, S, ldS, X, ldX, Y, ldY);
}

void cholsolve(const int m, const int n, const float* S,
    const int ldS, float* X, const int ldX, const float* Y, const int ldY) {
  cholsolve<float>(m, n, S, ldS, X, ldX, Y, ldY);
}

void inv(const int n, const double* A, const int ldA, double* B,
    const int ldB) {
  inv<double>(n, A, ldA, B, ldB);
}

void inv(const int n, const float* A, const int ldA, float* B,
    const int ldB) {
  inv<float>(n, A, ldA, B, ldB);
}

void cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB) {
  cholinv<double>(n, S, ldS, B, ldB);
}

void cholinv(const int n, const float* S, const int ldS, float* B,
    const int ldB) {
  cholinv<float>(n, S, ldS, B, ldB);
}

double ldet(const int n, const double* A, const int ldA) {
  return ldet<double>(n, A, ldA);
}

float ldet(const int n, const float* A, const int ldA) {
  return ldet<float>(n, A, ldA);
}

double lcholdet(const int n, const double* S, const int ldS) {
  return lcholdet<double>(n, S, ldS);
}

float lcholdet(const int n, const float* S, const int ldS) {
  return lcholdet<float>(n, S, ldS);
}

void transpose(const int m, const int n, const double x,
    const double* A, const int ldA, double* B, const int ldB) {
  transpose<double>(m, n, x, A, ldA, B, ldB);
}

void transpose(const int m, const int n, const float x,
    const float* A, const int ldA, float* B, const int ldB) {
  transpose<float>(m, n, x, A, ldA, B, ldB);
}

double trace(const int m, const int n, const double* A,
    const int ldA) {
  return trace<double>(m, n, A, ldA);
}

float trace(const int m, const int n, const float* A,
    const int ldA) {
  return trace<float>(m, n, A, ldA);
}

}
