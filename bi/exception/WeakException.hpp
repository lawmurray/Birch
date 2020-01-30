/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/type/NamedType.hpp"

namespace bi {
/**
 * Weak reference used on something other than a class type.
 *
 * @ingroup exception
 */
struct WeakException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Type expression.
   */
  WeakException(const NamedType* type);
};
}
