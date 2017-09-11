/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Incorrect type for condition in conditional or loop.
 *
 * @ingroup compiler_exception
 */
struct ConditionException: public CompilerException {
  /**
   * Constructor.
   */
  ConditionException(const Expression* o);
};
}
