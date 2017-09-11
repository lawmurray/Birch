/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Return.hpp"

namespace bi {
/**
 * Return value in function with no return type.
 *
 * @ingroup compiler_exception
 */
struct ReturnException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Return statement.
   */
  ReturnException(const Return* o);
};
}
