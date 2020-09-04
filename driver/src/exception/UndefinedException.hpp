/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"

namespace birch {
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

#include "src/generate/BirchGenerator.hpp"

template<class T>
birch::UndefinedException::UndefinedException(const T* o) {
  std::stringstream base;
  BirchGenerator buf(base, 0, true);
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
