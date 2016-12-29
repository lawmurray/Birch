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
  template<class ParameterType, class ReferenceType>
  AmbiguousReferenceException(const ReferenceType* ref,
      std::list<ParameterType*> matches);
};
}
