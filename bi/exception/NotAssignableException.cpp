/**
 * @file
 */
#include "bi/exception/NotAssignableException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::NotAssignableException::NotAssignableException(const Assignment* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: left side of assignment is not assignable\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';
  msg = base.str();
}
