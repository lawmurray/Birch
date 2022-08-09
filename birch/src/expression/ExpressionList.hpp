/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Expression list.
 *
 * @ingroup common
 */
class ExpressionList: public Expression {
public:
  /**
   * Constructor.
   *
   * @param head First in list.
   * @param tail Remaining list.
   * @param loc Location.
   */
  ExpressionList(Expression* head, Expression* tail, Location* loc = nullptr);

  virtual bool isTuple() const;

  virtual void accept(Visitor* visitor) const;

  /**
   * First element of list.
   */
  Expression* head;

  /**
   * Remainder of list.
   */
  Expression* tail;
};
}
