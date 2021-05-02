/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Empty expression.
 *
 * @ingroup expression
 */
class EmptyExpression: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  EmptyExpression(Location* loc = nullptr);

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;
};
}
