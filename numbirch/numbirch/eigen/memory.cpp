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

void array_init(ArrayControl* ctl, const size_t size) {
  assert(ctl);
  ctl->buf = malloc(size);
  ctl->size = size;
  ctl->streamAlloc = nullptr;
  ctl->streamWrite = nullptr;
}

void array_term(ArrayControl* ctl) {
  assert(ctl);
  free(ctl->buf, ctl->size);
}

void array_resize(ArrayControl* ctl, const size_t size) {
  ctl->buf = numbirch::realloc(ctl->buf, ctl->size, size);
  ctl->size = size;
}

void array_copy(ArrayControl* dst, const ArrayControl* src) {
  auto src1 = const_cast<ArrayControl*>(src);
  memcpy(dst->buf, src1->buf, std::min(dst->size, src1->size));
}

void array_wait(ArrayControl* ctl) {
  //
}

bool array_test(ArrayControl* ctl) {
  return true;
}

void before_read(ArrayControl* ctl) {
  //
}

void before_write(ArrayControl* ctl) {
  //
}

void after_read(ArrayControl* ctl) {
  //
}

void after_write(ArrayControl* ctl) {
  //
}

}
