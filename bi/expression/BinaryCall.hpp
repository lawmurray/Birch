/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Couple.hpp"

namespace bi {
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
