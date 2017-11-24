/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Return.hpp"
#include "bi/type/Type.hpp"

namespace bi {
/**
 * Incorrect type in return statement.
 *
 * @ingroup birch_exception
 */
struct ReturnTypeException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Return statement.
   * @param type Function return type.
   */
  ReturnTypeException(const Return* o, const Type* type);
};
}
