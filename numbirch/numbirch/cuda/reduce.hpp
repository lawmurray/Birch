/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/cuda/transform.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/common/get.hpp"
#include "numbirch/transform.hpp"
#include "numbirch/memory.hpp"

namespace numbirch {

template<class R, class T, class>
Array<R,0> count(const T& x) {
  ///@todo Avoid temporary
  return sum(transform(x, count_functor()));
}

template<class R, class T, class>
Array<R,0> sum(const T& x) {
  prefetch(x);
  Array<R,0> z;
  auto y = make_cub(x);
  void* tmp = nullptr;
  size_t bytes = 0;

  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, y, data(z), size(x),
      stream));
  tmp = device_malloc(bytes);
  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, y, data(z), size(x),
      stream));
  device_free(tmp);
  return z;
}

}
