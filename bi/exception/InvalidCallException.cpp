/**
 * @file
 */
#include "bi/exception/InvalidCallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::InvalidCallException::InvalidCallException(Type* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid argument types '" << o << "'\n";
  msg = base.str();
}
