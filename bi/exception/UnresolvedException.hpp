/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Unresolved identifier.
 *
 * @ingroup exception
 */
struct UnresolvedException: public Exception {
  /**
   * Constructor.
   */
  template<class ObjectType>
  UnresolvedException(const ObjectType* o);
};
}

#include "bi/io/bih_ostream.hpp"

template<class ObjectType>
bi::UnresolvedException::UnresolvedException(const ObjectType* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: unresolved identifier '" << o->name << "'\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
