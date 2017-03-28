/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Parenthesised expression.
 *
 * @ingroup compiler_common
 */
class Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param parens Parenthesised expression.
   */
  Parenthesised(Expression* parens = new EmptyExpression());

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * If these parentheses were constructed for a binary operator, get the
   * left operand. Otherwise undefined.
   */
  const Expression* getLeft() const;

  /**
   * If these parentheses were constructed for a binary operator, release the
   * left operand. Otherwise undefined.
   */
  Expression* releaseLeft();

  /**
   * If these parentheses were constructed for a binary or unary operator,
   * get the right operand. Otherwise undefined.
   */
  const Expression* getRight() const;

  /**
   * If these parentheses were constructed for a binary or unary operator,
   * release the right operand. Otherwise undefined.
   */
  Expression* releaseRight();

  /**
   * Expression in parentheses.
   */
  unique_ptr<Expression> parens;
};
}
