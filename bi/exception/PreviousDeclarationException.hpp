/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

namespace bi {
/**
 * Previous declaration.
 *
 * @ingroup compiler_exception
 */
struct PreviousDeclarationException: public CompilerException {
  /**
   * Constructor.
   */
  template<class ParameterType>
  PreviousDeclarationException(ParameterType* param, ParameterType* prev);
};
}
