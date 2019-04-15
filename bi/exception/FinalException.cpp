/**
 * @file
 */
#include "bi/exception/FinalException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::FinalException::FinalException(const Class* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: cannot inherit from a class with final qualifier\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
