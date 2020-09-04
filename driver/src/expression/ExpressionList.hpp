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

  /**
   * Destructor.
   */
  virtual ~ExpressionList();

  virtual bool isAssignable() const;
  virtual bool isTuple() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
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
