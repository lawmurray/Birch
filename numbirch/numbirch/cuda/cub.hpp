/**
 * @file
 * 
 * CUB integration.
 */
#pragma once

#include "numbirch/macro.hpp"
#include "numbirch/array.hpp"

#include <cub/cub.cuh>

namespace numbirch {
template<class T>
struct vector_element_functor {
  vector_element_functor(T* x, int incx) :
      x(x),
      incx(incx) {
    //
  }
  NUMBIRCH_HOST_DEVICE T& operator()(const int i) const {
    return x[i*incx];
  }
  T* x;
  int incx;
};

template<class T>
struct matrix_element_functor {
  matrix_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  NUMBIRCH_HOST_DEVICE T& operator()(const int i) const {
    int c = i/m;
    int r = i - c*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
static auto make_cub(Array<T,0>& x) {
  return data(x);
}

template<class T>
static auto make_cub(const Array<T,0>& x) {
  return data(x);
}

template<class T>
static auto make_cub(Array<T,1>& x) {
  auto elem = vector_element_functor<T>(data(x), stride(x));
  auto count = cub::CountingInputIterator<int>(0);
  return cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
}

template<class T>
static auto make_cub(const Array<T,1>& x) {
  auto elem = vector_element_functor<const T>(data(x), stride(x));
  auto count = cub::CountingInputIterator<int>(0);
  return cub::TransformInputIterator<const T,decltype(elem),decltype(count)>(
      count, elem);
}

template<class T>
static auto make_cub(Array<T,2>& A) {
  auto elem = matrix_element_functor<T>(data(A), rows(A), stride(A));
  auto count = cub::CountingInputIterator<int>(0);
  return cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
        count, elem);
}

template<class T>
static auto make_cub(const Array<T,2>& A) {
  auto elem = matrix_element_functor<const T>(data(A), rows(A), stride(A));
  auto count = cub::CountingInputIterator<int>(0);
  return cub::TransformInputIterator<const T,decltype(elem),decltype(count)>(
        count, elem);
}

}
