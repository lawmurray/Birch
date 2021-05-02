/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"
#include "src/common/Argumented.hpp"

namespace birch {
/**
 * Call to a function.
 *
 * @ingroup expression
 */
class Call: public Expression, public Single<Expression>, public Argumented {
public:
  /**
   * Constructor.
   *
   * @param single Expression indicating the function.
   * @param args Arguments.
   * @param loc Location.
   */
  Call(Expression* single, Expression* args, Location* loc = nullptr);

  /**
   * Constructor for call with no arguments.
   *
   * @param single Expression indicating the function.
   * @param loc Location.
   */
  Call(Expression* single, Location* loc = nullptr);

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
