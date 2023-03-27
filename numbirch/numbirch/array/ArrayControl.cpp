/**
 * @file
 */
#include "numbirch/array/ArrayControl.hpp"

#include "numbirch/memory.hpp"

#include <algorithm>

namespace numbirch {

ArrayControl::ArrayControl(const size_t bytes) :
    r(1) {
  array_init(this, bytes);
}

ArrayControl::ArrayControl(const ArrayControl& o) :
    r(1) {
  array_init(this, o.bytes);
  array_copy(this, &o);
}

ArrayControl::ArrayControl(const ArrayControl& o, const size_t bytes) :
    r(1) {
  array_init(this, bytes);
  array_copy(this, &o);
}

ArrayControl::~ArrayControl() {
  array_term(this);
}

bool ArrayControl::test() {
  return array_test(this);
}

void ArrayControl::realloc(const size_t bytes) {
  array_resize(this, bytes);
}

}
