/**
 * @file
 */
#include "numbirch/memory.hpp"
#include "numbirch/random.hpp"
#include "numbirch/eigen/eigen.hpp"

#include <cstdlib>
#include <cstring>

namespace numbirch {

void init() {
  /* older compiler versions that do not support thread-safe static local
   * variable initialization require Eigen to initialize such variables before
   * entering a parallel region */
  Eigen::initParallel();
  seed();
}

void wait() {
  //
}

void term() {
  //
}

void* malloc(const size_t size) {
  return std::malloc(size);
}

void* realloc(void* ptr, const size_t size) {
  return std::realloc(ptr, size);
}

void free(void* ptr) {
  std::free(ptr);
}

void memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height) {
  if (dpitch == width && spitch == width) {
    std::memcpy(dst, src, width*height);
  } else for (int i = 0; i < height; ++i) {
    std::memcpy((char*)dst + i*dpitch, (char*)src + i*spitch, width);
  }
}

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int>>
void memset(void* dst, const size_t dpitch, const T value, const size_t width,
    const size_t height) {
  auto A = (T*)dst;
  for (int i = 0; i < width/sizeof(T); ++i) {
    for (int j = 0; j < height; ++j) {
      A[i + j*dpitch/sizeof(T)] = value;
    }
  }
}

template void memset(void*, const size_t, const double, const size_t,
    const size_t);
template void memset(void*, const size_t, const float, const size_t,
    const size_t);
template void memset(void*, const size_t, const int, const size_t,
    const size_t);
template void memset(void*, const size_t, const bool, const size_t,
    const size_t);

}
