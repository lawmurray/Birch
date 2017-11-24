/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Parenthesised expression.
 *
 * @ingroup birch_common
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
