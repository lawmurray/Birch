/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/expression/EmptyExpression.hpp"

namespace birch {
/**
 * Object with brackets.
 *
 * @ingroup common
 */
class Bracketed {
public:
  /**
   * Constructor.
   *
   * @param brackets Expression in square brackets.
   */
  Bracketed(Expression* brackets);

  /**
   * Square bracket expression.
   */
  Expression* brackets;
};
}
