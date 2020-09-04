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
   * Destructor.
   */
  virtual ~Bracketed() = 0;

  /**
   * Square bracket expression.
   */
  Expression* brackets;
};
}
