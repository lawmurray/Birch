/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Call to a unary operator.
 *
 * @ingroup expression
 */
class UnaryCall: public Expression, public Named, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param name Operator name (symbol).
   * @param single Operand.
   * @param loc Location.
   */
  UnaryCall(Name* name, Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
