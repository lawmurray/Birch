/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Incorrect use of "this" keyword.
 *
 * @ingroup compiler_exception
 */
struct ThisException: public CompilerException {
  /**
   * Constructor.
   */
  ThisException(const Expression* expr);
};
}
