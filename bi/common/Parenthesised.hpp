/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
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
  Parenthesised(Expression* parens = new EmptyExpression());

  /**
   * Destructor.
   */
  virtual ~Parenthesised() = 0;

  /**
   * Expression in parentheses.
   */
  unique_ptr<Expression> parens;
};
}
