/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Call.hpp"

namespace bi {
/**
 * Call on something that is not of function type.
 *
 * @ingroup birch_exception
 */
struct NotFunctionException: public CompilerException {
  /**
   * Constructor.
   */
  NotFunctionException(Call* o);
};
}
