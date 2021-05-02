/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Couple.hpp"

namespace birch {
/**
 * Range expression.
 *
 * @ingroup expression
 */
class Range: public Expression, public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  Range(Expression* left, Expression* right, Location* loc =
      nullptr);

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
