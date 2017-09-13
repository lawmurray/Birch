/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Expression in parentheses.
 *
 * @ingroup compiler_expression
 */
class Parentheses: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in parentheses.
   * @param loc Location.
   */
  Parentheses(Expression* single, Location* loc = nullptr);

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
