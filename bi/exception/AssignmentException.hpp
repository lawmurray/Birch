/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Assign.hpp"

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
  AssignmentException(const Assign* o);
};
}
