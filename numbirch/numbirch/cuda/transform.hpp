/**
 * @file
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/array.hpp"
#include "numbirch/macro.hpp"
#include "numbirch/memory.hpp"
#include "numbirch/type.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {
/*
 * Prefetch an array onto device.
 */
template<class T, int D>
void prefetch(const Array<T,D>& x) {
  CUDA_CHECK(cudaMemPrefetchAsync(x.data(), x.volume()*sizeof(T), device,
      stream));
}

/*
 * Prefetch a scalar onto device---null operation.
 */
template<class T, class = std::enable_if_t<is_arithmetic<T>::value,int>>
void prefetch(const T& x) {
  //
}

/*
 * Unary transform.
 */
template<class T, class R, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, R B, const int ldB, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(B, i, j, ldB) = f(element(A, i, j, ldA));
  }
}
template<class T, class Functor>
auto transform(const T& x, Functor f) {
  using R = decltype(f(value_t<T>()));
  constexpr int D = dimension_v<T>;
  auto y = Array<R,D>(shape(x));
  auto m = width(x);
  auto n = height(x);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), f);
  return y;
}

/*
 * Gradient of unary transform.
 */
template<class G, class T, class Functor>
auto transform_grad(const G& g, const T& x, Functor f) {
  return transform(g, x, f);  // same as binary transform
}

/*
 * Binary transform.
 */
template<class T, class U, class R, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, const U B, const int ldB, R C, const int ldC,
    Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(C, i, j, ldC) = f(element(A, i, j, ldA), element(B, i, j, ldB));
  }
}
template<class T, class U, class Functor>
auto transform(const T& x, const U& y, Functor f) {
  using R = decltype(f(value_t<T>(),value_t<U>()));
  constexpr int D = std::max(dimension_v<T>, dimension_v<U>);
  auto m = std::max(width(x), width(y));
  auto n = std::max(height(x), height(y));
  auto z = Array<R,D>(make_shape<D>(m, n));
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), data(z), stride(z), f);
  return z;
}

/*
 * Gradient of binary transform.
 */
template<class G, class T, class U, class V, class W, class Functor>
__global__ void kernel_transform_grad(const int m, const int n, const G g,
    const int ldg, const T A, const int ldA, const U B, const int ldB,
    V GA, const int ldGA, W GB, const int ldGB, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    auto pair = f(element(g, i, j, ldg), element(A, i, j, ldA),
        element(B, i, j, ldB));
    element(GA, i, j, ldGA) = pair.first;
    element(GB, i, j, ldGB) = pair.second;
  }
}
template<class G, class T, class U, class Functor>
auto transform_grad(const G& g, const T& x, const U& y, Functor f) {
  using P = decltype(f(value_t<G>(),value_t<T>(),value_t<U>()));
  using V = typename P::first_type;
  using W = typename P::second_type;
  constexpr int D = std::max(std::max(dimension_v<G>, dimension_v<T>),
      dimension_v<U>);
  auto m = std::max(std::max(width(g), width(x)), width(y));
  auto n = std::max(std::max(height(g), height(x)), height(y));
  auto a = Array<V,D>(make_shape<D>(m, n));
  auto b = Array<W,D>(make_shape<D>(m, n));
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform_grad<<<grid,block,0,stream>>>(m, n, data(g), stride(g),
      data(x), stride(x), data(y), stride(y), data(a), stride(a), data(b),
      stride(b), f);
  return std::make_pair(a, b);
}

/*
 * Ternary transform.
 */
template<class T, class U, class V, class R, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, const U B, const int ldB, const V C, const int ldC,
    R D, const int ldD, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(D, i, j, ldD) = f(element(A, i, j, ldA), element(B, i, j, ldB),
        element(C, i, j, ldC));
  }
}
template<class T, class U, class V, class Functor>
auto transform(const T& x, const U& y, const V& z, Functor f) {
  using R = decltype(f(value_t<T>(),value_t<U>(),value_t<V>()));
  constexpr int D = std::max(std::max(dimension_v<T>, dimension_v<U>),
      dimension_v<V>);
  auto m = std::max(std::max(width(x), width(y)), width(z));
  auto n = std::max(std::max(height(x), height(y)), height(z));
  auto a = Array<R,D>(make_shape<D>(m, n));
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), data(z), stride(z), data(a), stride(a), f);
  return a;
}

}
