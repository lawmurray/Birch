/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"
#include "src/expression/EmptyExpression.hpp"

namespace birch {
/**
 * Expression in parentheses.
 *
 * @ingroup expression
 */
class Parentheses: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in parentheses.
   * @param loc Location.
   */
  Parentheses(Expression* single, Location* loc = nullptr);

  virtual const Expression* strip() const;
  virtual bool isSlice() const;
  virtual bool isTuple() const;
  virtual bool isMembership() const;

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
