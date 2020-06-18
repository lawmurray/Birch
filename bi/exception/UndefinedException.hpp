/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Undefined identifier.
 *
 * @ingroup exception
 */
struct UndefinedException: public Exception {
  /**
   * Constructor.
   */
  template<class T>
  UndefinedException(const T* o);
};
}

#include "bi/io/bih_ostream.hpp"

template<class T>
bi::UndefinedException::UndefinedException(const T* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: '" << o->name << "' is undefined\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
