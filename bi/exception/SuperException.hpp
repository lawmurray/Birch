/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Use of "super" keyword outside of a class.
 *
 * @ingroup compiler_exception
 */
struct SuperException: public CompilerException {
  /**
   * Constructor.
   */
  SuperException(const Expression* expr);
};
}
