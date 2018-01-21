/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Assignment.hpp"

namespace bi {
/**
 * Invalid use of assignment operator.
 *
 * @ingroup exception
 */
struct AssignmentException: public CompilerException {
  /**
   * Constructor.
   */
  AssignmentException(const Assignment* o);
};
}
