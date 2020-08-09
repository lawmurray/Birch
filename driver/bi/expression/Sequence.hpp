/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
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
