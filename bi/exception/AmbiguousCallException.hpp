/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Parameterised.hpp"

namespace bi {
/**
 * Ambiguous overloaded function call.
 *
 * @ingroup exception
 */
struct AmbiguousCallException: public CompilerException {
  /**
   * Constructor.
   */
  AmbiguousCallException(const Argumented* o,
      const std::set<Parameterised*>& matches);
};
}
