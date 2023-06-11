/**
 * @file
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/common/reduce.inl"
#include "numbirch/common/transform.inl"
#include "numbirch/cuda/transform.inl"

namespace numbirch {

template<numeric T>
Array<int,0> count(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return count_functor()(x);
  } else if (size(x) == 0) {
    return 0;
  } else {
    ///@todo Avoid temporary
    return sum(transform(x, count_functor()));
  }
}

template<numeric T>
Array<value_t<T>,0> sum(const T& x) {
  if constexpr (is_scalar_v<T>) {
    return x;
  } else if (size(x) == 0) {
    return value_t<T>(0);
  } else {
    prefetch(x);
    Array<value_t<T>,0> z;
    auto y = make_cub(x);
    void* tmp = nullptr;
    value_t<T>* dst = nullptr;
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, y, dst, size(x),
        stream));
    tmp = device_malloc(bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, y, buffer(z), size(x),
        stream));
    device_free(tmp, bytes);
    return z;
  }
}

}
