/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Cast.hpp"

namespace bi {
/**
 * Invalid use of the cast operator.
 *
 * @ingroup compiler_exception
 */
struct CastException: public CompilerException {
  /**
   * Constructor.
   */
  CastException(const Cast* o);
};
}
