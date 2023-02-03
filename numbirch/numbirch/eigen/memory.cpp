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

void* realloc(void* oldptr, const size_t oldsize, const size_t newsize) {
  return std::realloc(oldptr, newsize);
}

void free(void* ptr) {
  std::free(ptr);
}

void free(void* ptr, const size_t size) {
  std::free(ptr);
}

void memcpy(void* dst, const void* src, size_t n) {
  std::memcpy(dst, src, n);
}

void* event_create() {
  return 0;
}

void event_destroy(void* evt) {
  //
}

bool event_test(void* evt) {
  return true;
}

void event_wait(void* evt) {
  //
}

void before_read(const ArrayControl* ctl) {
  //
}

void before_write(const ArrayControl* ctl) {
  //
}

void after_read(const ArrayControl* ctl) {
  //
}

void after_write(const ArrayControl* ctl) {
  //
}

}
