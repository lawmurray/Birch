/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

#include <list>

namespace bi {
/**
 * Ambiguous reference in program.
 *
 * @ingroup compiler_exception
 */
struct AmbiguousReferenceException: public CompilerException {
  /**
   * Constructor.
   */
  template<class ReferenceType>
  AmbiguousReferenceException(ReferenceType* ref);
};
}

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ReferenceType>
bi::AmbiguousReferenceException::AmbiguousReferenceException(
    ReferenceType* ref) {
  std::stringstream base;
  bih_ostream buf(base);
  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "error: ambiguous reference '" << ref->name << "'\n";
  buf << ref << '\n';
  for (auto iter = ref->matches.begin(); iter != ref->matches.end(); ++iter) {
    if ((*iter)->loc) {
      buf << (*iter)->loc;
    }
    buf << "note: candidate\n";
    buf << *iter << '\n';
  }
  msg = base.str();
}
