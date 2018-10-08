/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Call on something that is not of function type.
 *
 * @ingroup exception
 */
struct NotFunctionException: public CompilerException {
  /**
   * Constructor.
   */
  NotFunctionException(Expression* o);
};
}
