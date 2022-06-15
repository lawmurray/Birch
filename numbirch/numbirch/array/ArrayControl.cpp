/**
 * @file
 */
#include "numbirch/array/ArrayControl.hpp"

#include "numbirch/memory.hpp"

#include <algorithm>

namespace numbirch {

ArrayControl::ArrayControl(const size_t bytes) :
    buf(malloc(bytes)),
    readEvt(event_create()),
    writeEvt(event_create()),
    bytes(bytes),
    r(1) {
  event_record_write(writeEvt);
}

ArrayControl::ArrayControl(const ArrayControl& o) :
    buf(malloc(o.bytes)),
    readEvt(event_create()),
    writeEvt(event_create()),
    bytes(o.bytes),
    r(1) {
  event_join(o.writeEvt);
  memcpy(buf, o.buf, o.bytes);
  event_record_read(o.readEvt);
  event_record_write(writeEvt);
}

ArrayControl::ArrayControl(const ArrayControl& o, const size_t bytes) :
    buf(malloc(bytes)),
    readEvt(event_create()),
    writeEvt(event_create()),
    bytes(bytes),
    r(1) {
  event_join(o.writeEvt);
  memcpy(buf, o.buf, std::min(bytes, o.bytes));
  event_record_read(o.readEvt);
  event_record_write(writeEvt);
}

ArrayControl::~ArrayControl() {
  event_join(readEvt);
  event_join(writeEvt);
  free(buf, bytes);
  event_destroy(readEvt);
  event_destroy(writeEvt);
}

bool ArrayControl::test() {
  return event_test(readEvt) && event_test(writeEvt);
}

void ArrayControl::realloc(const size_t bytes) {
  event_join(readEvt);
  event_join(writeEvt);
  buf = numbirch::realloc(buf, this->bytes, bytes);
  this->bytes = bytes;
  event_record_write(writeEvt);
}

}
