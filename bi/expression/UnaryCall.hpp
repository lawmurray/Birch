/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/common/Argumented.hpp"

namespace bi {
/**
 * Call to a unary operator.
 *
 * @ingroup compiler_expression
 */
class UnaryCall: public Expression,
    public Single<Expression>,
    public Argumented {
public:
  /**
   * Constructor.
   *
   * @param single Expression indicating the function.
   * @param args Arguments.
   * @param loc Location.
   */
  UnaryCall(Expression* single, Expression* args, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
