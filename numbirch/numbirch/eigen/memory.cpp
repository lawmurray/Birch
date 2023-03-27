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

void array_init(ArrayControl* ctl, const size_t bytes) {
  assert(ctl);
  ctl->buf = malloc(bytes);
  ctl->bytes = bytes;
  ctl->streamAlloc = nullptr;
  ctl->streamWrite = nullptr;
}

void array_term(ArrayControl* ctl) {
  assert(ctl);
  free(ctl->buf, ctl->bytes);
}

void array_resize(ArrayControl* ctl, const size_t bytes) {
  ctl->buf = numbirch::realloc(ctl->buf, ctl->bytes, bytes);
  ctl->bytes = bytes;
}

void array_copy(ArrayControl* dst, const ArrayControl* src) {
  auto src1 = const_cast<ArrayControl*>(src);
  memcpy(dst->buf, src1->buf, std::min(dst->bytes, src1->bytes));
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
