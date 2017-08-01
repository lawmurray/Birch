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
  if (o->isBinary()) {
    buf << "error: no overload for argument types '" << o->getLeft();
    buf << "' and '" << o->getRight() << "'\n";
  } else {
    buf << "error: no overload for argument types '" << o << "'\n";
  }

  msg = base.str();
}
