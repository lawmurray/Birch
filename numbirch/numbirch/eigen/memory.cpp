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

void* record() {
  return 0;
}

void wait(void* evt) {
  //
}

void forget(void* evt) {
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

void free(void* ptr, const size_t size) {
  std::free(ptr);
}

}
