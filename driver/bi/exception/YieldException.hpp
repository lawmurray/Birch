/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"
#include "bi/statement/Yield.hpp"

namespace bi {
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
