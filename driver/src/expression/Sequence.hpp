/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"
#include "src/expression/EmptyExpression.hpp"

namespace birch {
/**
 * Sequence expression.
 *
 * @ingroup expression
 */
class Sequence: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in brackets.
   * @param loc Location.
   */
  Sequence(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Sequence();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
