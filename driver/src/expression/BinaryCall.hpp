/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Named.hpp"
#include "src/common/Couple.hpp"

namespace birch {
/**
 * Call to a binary operator.
 *
 * @ingroup expression
 */
class BinaryCall:
    public Expression,
    public Named,
    public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  BinaryCall(Expression* left, Name* name, Expression* right,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
