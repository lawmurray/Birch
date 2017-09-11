/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Yield.hpp"
#include "bi/type/Type.hpp"

namespace bi {
/**
 * Incorrect type in yield statement.
 *
 * @ingroup compiler_exception
 */
struct YieldTypeException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Yield statement.
   * @param type Fiber yield type.
   */
  YieldTypeException(const Yield* o, const Type* type);
};
}
