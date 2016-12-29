/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Incorrect use of '.' operator
 *
 * @ingroup compiler_exception
 */
struct TraversalException: public CompilerException {
  /**
   * Constructor.
   */
  TraversalException(const Expression* expr);
};
}
