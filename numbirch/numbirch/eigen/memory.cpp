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

void term() {
  //
}

void* malloc(const size_t bytes) {
  return std::malloc(bytes);
}

void* realloc(void* oldptr, const size_t oldbytes, const size_t newbytes) {
  return std::realloc(oldptr, newbytes);
}

void free(void* ptr) {
  std::free(ptr);
}

void free(void* ptr, const size_t bytes) {
  std::free(ptr);
}

void memcpy(void* dst, const void* src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void* stream_get() {
  return 0;
}

void stream_wait(void* s) {
  //
}

void stream_join(void* s) {
  //
}

void stream_finish(void* streamAlloc, void* stream) {
  //
}

void lock() {
  //
}

void unlock() {
  //
}

void lock_shared() {
  //
}

void unlock_shared() {
  //
}

}
