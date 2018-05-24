/**
 * @file
 */
#include "bi/exception/CastException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::CastException::CastException(const Cast* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: only objects can be cast\n";
  buf << "note: operand has type '" << o->single->type << "'\n";
  msg = base.str();
}
