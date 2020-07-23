/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Redefined identifier.
 *
 * @ingroup exception
 */
struct RedefinedException: public Exception {
  /**
   * Constructor.
   *
   * @param o Redefinition.
   * @param prev Previous definition.
   */
  template<class T, class U>
  RedefinedException(const T* o, const U* prev);
};
}

#include "bi/io/bih_ostream.hpp"

template<class T, class U>
bi::RedefinedException::RedefinedException(const T* o, const U* prev) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: '" << o->name << "' is already defined\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';
  if (prev->loc) {
    buf << prev->loc;
  }
  buf << "note: previous definition is here\n";
  buf << prev << '\n';

  msg = base.str();
}
