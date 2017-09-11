/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Yield.hpp"

namespace bi {
/**
 * Yield statement outside of fiber.
 *
 * @ingroup compiler_exception
 */
struct YieldException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Yield statement.
   */
  YieldException(const Yield* o);
};
}
