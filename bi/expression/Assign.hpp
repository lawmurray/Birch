/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Couple.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Assignment expression.
 *
 * @ingroup statement
 */
class Assign: public Expression,
    public Named,
    public Couple<Expression>,
    public Reference<AssignmentOperator> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param op Operator.
   * @param right Right operand.
   * @param loc Location.
   * @param target Target.
   */
  Assign(Expression* left, Name* op, Expression* right, Location* loc =
      nullptr, AssignmentOperator* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Assign();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
