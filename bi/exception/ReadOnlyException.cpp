/**
 * @file
 */
#include "bi/exception/ReadOnlyException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::ReadOnlyException::ReadOnlyException(const Type* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: read-only types must be used for global variables, closed fiber parameters and closed fiber yields.\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;
  msg = base.str();
}
