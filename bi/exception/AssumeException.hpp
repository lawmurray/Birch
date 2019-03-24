/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Assume.hpp"

namespace bi {
/**
 * Invalid use of assume statement.
 *
 * @ingroup exception
 */
struct AssumeException: public CompilerException {
  /**
   * Constructor.
   */
  AssumeException(const Assume* o);
};
}
