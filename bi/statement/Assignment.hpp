/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Couple.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to assignment operator.
 *
 * @ingroup statement
 */
class Assignment: public Statement,
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
  Assignment(Expression* left, Name* op, Expression* right, Location* loc =
      nullptr, AssignmentOperator* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Assignment();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
