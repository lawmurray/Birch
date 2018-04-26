/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/AssignmentOperator.hpp"

namespace bi {
/**
 * Invalid type for assignment operator declaration.
 *
 * @ingroup exception
 */
struct AssignmentOperatorException: public CompilerException {
  /**
   * Constructor.
   */
  AssignmentOperatorException(const AssignmentOperator* o);
};
}
