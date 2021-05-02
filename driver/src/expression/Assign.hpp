/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/statement/AssignmentOperator.hpp"
#include "src/common/Named.hpp"
#include "src/common/Couple.hpp"

namespace birch {
/**
 * Expression using `<-`, `<~` or `~>` operator.
 *
 * @ingroup expression
 */
class Assign: public Expression,
    public Named,
    public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param op Operator.
   * @param right Right operand.
   * @param loc Location.
   */
  Assign(Expression* left, Name* op, Expression* right, Location* loc =
      nullptr);

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
