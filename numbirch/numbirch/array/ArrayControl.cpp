/**
 * @file
 */
#include "numbirch/array/ArrayControl.hpp"

#include "numbirch/memory.hpp"

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
    ArrayControl(o.bytes) {
  event_join(o.writeEvt);
  memcpy(buf, o.buf, bytes);
  event_record_write(writeEvt);
}

ArrayControl::ArrayControl(const ArrayControl& o, const size_t bytes) :
    ArrayControl(bytes) {
  event_join(o.writeEvt);
  memcpy(buf, o.buf, std::min(bytes, o.bytes));
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
