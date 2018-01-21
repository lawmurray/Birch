/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Program option with unsupported type.
 *
 * @ingroup exception
 */
struct UnsupportedOptionTypeException: public CompilerException {
  /**
   * Constructor.
   */
  UnsupportedOptionTypeException(Type* type);
};
}
