/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/type/Type.hpp"

#include <list>

namespace bi {
/**
 * Invalid function call.
 *
 * @ingroup compiler_exception
 */
struct InvalidCallException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o The invalid call.
   * @param available The available overloads.
   */
  InvalidCallException(Type* o,
      const std::list<Type*>& available = std::list<Type*>());
};
}
