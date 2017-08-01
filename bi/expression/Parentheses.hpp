/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
class Parameter;

/**
 * Expression in parentheses.
 *
 * @ingroup compiler_expression
 */
class Parentheses: public Expression, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in parentheses.
   * @param loc Location.
   */
  Parentheses(Expression* single = new EmptyExpression(),
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Parentheses();

  virtual Expression* strip();
  virtual Iterator<Expression> begin() const;
  virtual Iterator<Expression> end() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
