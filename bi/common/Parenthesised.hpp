/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/primitive/unique_ptr.hpp"

#include <cassert>

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
  Parenthesised(Expression* parens);

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
   * If these parentheses were constructed for a binary operator, get the
   * right operand. Otherwise undefined.
   */
  const Expression* getRight() const;

  /**
   * As getLeft(), but releases ownership.
   */
  Expression* releaseLeft();

  /**
   * As getRight(), but releases ownership.
   */
  Expression* releaseRight();

  /**
   * Expression in parentheses.
   */
  unique_ptr<Expression> parens;
};
}
