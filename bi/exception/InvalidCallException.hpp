/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Call.hpp"

namespace bi {
/**
 * Invalid function call.
 *
 * @ingroup compiler_exception
 */
struct InvalidCallException: public CompilerException {
  /**
   * Constructor.
   */
  InvalidCallException(Call* o);
};
}
