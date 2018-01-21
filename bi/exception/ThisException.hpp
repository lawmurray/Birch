/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Use of "this" keyword outside of a class.
 *
 * @ingroup exception
 */
struct ThisException: public CompilerException {
  /**
   * Constructor.
   */
  ThisException(const Expression* expr);
};
}
