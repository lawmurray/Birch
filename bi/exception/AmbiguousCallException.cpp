/**
 * @file
 */
#include "bi/exception/AmbiguousCallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::AmbiguousCallException::AmbiguousCallException(const Call* o,
    const std::list<Type*>& matches) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: ambiguous call '" << o << "'\n";
  buf << o << '\n';
  for (auto match : matches) {
    if (match->loc) {
      buf << match->loc;
    }
    buf << "note: candidate\n";
    buf << match << '\n';
  }
  msg = base.str();
}
