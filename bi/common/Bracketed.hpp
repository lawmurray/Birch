/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
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
