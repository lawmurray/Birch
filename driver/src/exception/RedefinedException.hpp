/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"

namespace birch {
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

#include "src/generate/BirchGenerator.hpp"

template<class T, class U>
birch::RedefinedException::RedefinedException(const T* o, const U* prev) {
  std::stringstream base;
  BirchGenerator buf(base, 0, true);
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
