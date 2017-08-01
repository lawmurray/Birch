/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Couple.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Membership operator expression.
 *
 * @ingroup compiler_expression
 */
class Member: public Expression, public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  Member(Expression* left, Expression* right, Location* loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Member();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
