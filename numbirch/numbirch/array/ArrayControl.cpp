/**
 * @file
 */
#include "numbirch/array/ArrayControl.hpp"

#include "numbirch/memory.hpp"

#include <algorithm>

namespace numbirch {

ArrayControl::ArrayControl(const size_t bytes) :
    buf(malloc(bytes)),
    evt(event_create()),
    bytes(bytes),
    r(1) {
  after_write(this);
}

ArrayControl::ArrayControl(const ArrayControl& o) :
    buf(malloc(o.bytes)),
    evt(event_create()),
    bytes(o.bytes),
    r(1) {
  before_read(&o);
  before_write(this);
  memcpy(buf, o.buf, o.bytes);
  after_write(this);
  after_read(&o);
}

ArrayControl::ArrayControl(const ArrayControl& o, const size_t bytes) :
    buf(malloc(bytes)),
    evt(event_create()),
    bytes(bytes),
    r(1) {
  before_read(&o);
  before_write(this);
  memcpy(buf, o.buf, std::min(bytes, o.bytes));
  after_write(this);
  after_read(&o);
}

ArrayControl::~ArrayControl() {
  before_write(this);
  free(buf, bytes);
  event_destroy(evt);
}

bool ArrayControl::test() {
  return event_test(evt);
}

void ArrayControl::realloc(const size_t bytes) {
  before_write(this);
  buf = numbirch::realloc(buf, this->bytes, bytes);
  this->bytes = bytes;
  after_write(this);
}

}
