/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

namespace bi {
/**
 * Ambiguous overloaded function call.
 *
 * @ingroup compiler_exception
 */
struct AmbiguousCallException: public CompilerException {
  /**
   * Constructor.
   */
  template<class ObjectType>
  AmbiguousCallException(ObjectType* o);
};
}

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ObjectType>
bi::AmbiguousCallException::AmbiguousCallException(ObjectType* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: ambiguous call '" << o << "'\n";
  buf << o << '\n';
  for (auto match : o->matches) {
    if (match->loc) {
      buf << match->loc;
    }
    buf << "note: candidate\n";
    buf << match << '\n';
  }
  msg = base.str();
}
