/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/expression/EmptyExpression.hpp"

namespace birch {
/**
 * Parenthesised expression.
 *
 * @ingroup common
 */
class Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param parens Parenthesised expression.
   */
  Parenthesised(Expression* parens);

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * Expression in parentheses.
   */
  Expression* parens;
};
}
