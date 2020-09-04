/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"
#include "src/statement/Yield.hpp"

namespace birch {
/**
 * Yield outside fiber exception.
 *
 * @ingroup exception
 */
struct YieldException: public Exception {
  /**
   * Constructor.
   */
  YieldException(const Yield* o);
};
}
