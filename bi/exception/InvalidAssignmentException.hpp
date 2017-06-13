/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Initial value of a variable is not of a compatible type.
 *
 * @ingroup compiler_exception
 */
struct InvalidAssignmentException: public CompilerException {
  /**
   * Constructor.
   */
  InvalidAssignmentException(const Expression* expr);
};
}
