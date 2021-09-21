/**
 * @file
 */
#include "numbirch/numbirch.hpp"
#include "numbirch/default/numbirch.hpp"

void numbirch::init() {
  /* older compiler versions that do not support thread-safe static local
   * variable initialization require Eigen to initialize such variables before
   * entering a parallel region */
  Eigen::initParallel();
}

void numbirch::term() {
  //
}

void* numbirch::malloc(const size_t size) {
  return std::malloc(size);
}

void* numbirch::realloc(void* ptr, const size_t size) {
  return std::realloc(ptr, size);
}

void numbirch::free(void* ptr) {
  std::free(ptr);
}

void numbirch::memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height) {
  if (dpitch == width && spitch == width) {
    std::memcpy(dst, src, width*height);
  } else for (int i = 0; i < height; ++i) {
    std::memcpy((char*)dst + i*dpitch, (char*)src + i*spitch, width);
  }
}

void numbirch::wait() {
  //
}
