/**
 * @file
 * 
 * CUB integration.
 */
#pragma once

#include "numbirch/macro.hpp"

#include <cub/cub.cuh>

namespace numbirch {
/*
 * Pair of iterators defining a range.
 */
template<class Iterator>
struct cub_range {
  cub_range(Iterator first, Iterator second) :
      first(first),
      second(second) {
    //
  }
  auto begin() {
    return first;
  }
  auto end() {
    return second;
  }
  Iterator first;
  Iterator second;
};

template<class Iterator>
static auto make_cub_range(Iterator first, Iterator second) {
  return cub_range<Iterator>(first, second);
}

template<class T>
static auto make_cub(Array<T,0>& x) {
  auto begin = data(x);
  return make_cub_range(begin, begin + size(x));
}

template<class T>
static auto make_cub(const Array<T,0>& x) {
  auto begin = data(x);
  return make_cub_range(begin, begin + size(x));
}

template<class T>
struct vector_element_functor {
  vector_element_functor(T* x, int incx) :
      x(x),
      incx(incx) {
    //
  }
  DEVICE T operator()(const int i) const {
    return x[i*incx];
  }
  T* x;
  int incx;
};

template<class T>
static auto make_cub(Array<T,1>& x) {
  auto elem = vector_element_functor<T>(data(x), stride(x));
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),
      decltype(count)>(count, elem);
  return make_cub_range(begin, begin + size(x));
}

template<class T>
static auto make_cub(const Array<T,1>& x) {
  auto elem = vector_element_functor<const T>(data(x), stride(x));
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<const T,decltype(elem),
      decltype(count)>(count, elem);
  return make_cub_range(begin, begin + size(x));
}

template<class T>
struct matrix_element_functor {
  matrix_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  DEVICE T operator()(const int i) const {
    int c = i/m;
    int r = i - c*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
static auto make_cub(Array<T,2>& A) {
  auto elem = matrix_element_functor<T>(data(A), rows(A), stride(A));
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),
      decltype(count)>(count, elem);
  return make_cub_range(begin, begin + size(A));
}

template<class T>
static auto make_cub(const Array<T,2>& A) {
  auto elem = matrix_element_functor<const T>(data(A), rows(A), stride(A));
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<const T,decltype(elem),
      decltype(count)>(count, elem);
  return make_cub_range(begin, begin + size(A));
}

}
