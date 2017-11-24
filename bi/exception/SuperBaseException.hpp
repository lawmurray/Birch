/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Use of "super" keyword outside of a class with a base.
 *
 * @ingroup birch_exception
 */
struct SuperBaseException: public CompilerException {
  /**
   * Constructor.
   */
  SuperBaseException(const Expression* expr);
};
}
