/**
 * @file
 */
#include "bi/exception/FiberTypeException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::FiberTypeException::FiberTypeException(const Statement* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: yield type must be a fiber type\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
