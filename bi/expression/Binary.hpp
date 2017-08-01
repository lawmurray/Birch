/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Couple.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Operands to a binary operator.
 *
 * @ingroup compiler_expression
 */
class Binary: public Expression, public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  Binary(Expression* left, Expression* right, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Binary();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
