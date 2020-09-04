/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Named.hpp"
#include "src/common/Couple.hpp"
#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Statement using `~` or `<-?` operator.
 *
 * @ingroup statement
 */
class Assume: public Statement, public Named, public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param op Operator.
   * @param right Right operand.
   * @param loc Location.
   */
  Assume(Expression* left, Name* op, Expression* right, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Assume();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
