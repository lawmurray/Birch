/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Named.hpp"
#include "src/common/Single.hpp"

namespace birch {
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
