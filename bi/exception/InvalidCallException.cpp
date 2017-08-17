/**
 * @file
 */
#include "bi/exception/InvalidCallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::InvalidCallException::InvalidCallException(Type* o,
    const std::list<Type*>& available) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: no overload for argument types '" << o << "'\n";
  for (auto overload : available) {
    if (overload->loc) {
      buf << overload->loc;
    }
    buf << "note: candidate\n";
    buf << overload << '\n';
  }

  msg = base.str();
}
