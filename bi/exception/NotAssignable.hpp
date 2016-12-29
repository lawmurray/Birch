/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Left hand side of assignment operator is not assignable.
 *
 * @ingroup compiler_exception
 */
struct NotAssignable: public CompilerException {
  /**
   * Constructor.
   */
  NotAssignable(const Expression* expr);
};
}
