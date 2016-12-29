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
