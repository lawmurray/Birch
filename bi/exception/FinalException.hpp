/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
/**
 * Attempt to inherit from a class qualified as final.
 *
 * @ingroup exception
 */
struct FinalException: public CompilerException {
  /**
   * Constructor.
   */
  FinalException(const Class* o);
};
}
