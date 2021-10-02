/**
 * @file
 * 
 * CUB integration.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/common/functor.hpp"

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
static auto make_cub_vector(T* x, const int n, const int incx) {
  auto elem = vector_element_functor<T>(x, incx);
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
  return make_cub_range(begin, begin + n);
}

template<class T>
static auto make_cub_matrix(T* A, const int m, const int n, const int ldA) {
  auto elem = matrix_element_functor<T>(A, m, ldA);
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
  return make_cub_range(begin, begin + m*n);
}

template<class T>
static auto make_cub_matrix_transpose(T* A, const int m, const int n,
    const int ldA) {
  auto elem = matrix_transpose_element_functor<T>(A, m, ldA);
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
  return make_cub_range(begin, begin + m*n);
}

}
