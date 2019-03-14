/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Couple.hpp"

namespace bi {
/**
 * Reference to assignment operator.
 *
 * @ingroup statement
 */
class Assume: public Statement, public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   * @param target Target.
   */
  Assume(Expression* left, Expression* right, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Assume();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
