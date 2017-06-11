/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

namespace bi {
/**
 * Unresolved reference in program.
 *
 * @ingroup compiler_exception
 */
struct UnresolvedReferenceException: public Exception {
  /**
   * Constructor.
   */
  template<class ReferenceType>
  UnresolvedReferenceException(const ReferenceType* ref);
};
}

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ReferenceType>
bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const ReferenceType* ref) {
  std::stringstream base;
  bih_ostream buf(base);
  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "error: unresolved reference '" << ref->name << "'\n";
  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "note: in\n";
  buf << ref << '\n';

  msg = base.str();
}
