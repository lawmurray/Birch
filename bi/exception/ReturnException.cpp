/**
 * @file
 */
#include "bi/exception/ReturnException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ReturnException::ReturnException(const Return* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: return value, but function has no return type\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
