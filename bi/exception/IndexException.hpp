/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Incorrect type for array or loop index.
 *
 * @ingroup birch_exception
 */
struct IndexException: public CompilerException {
  /**
   * Constructor.
   */
  IndexException(const Expression* o);
};
}
