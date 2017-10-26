/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Expression list.
 *
 * @ingroup compiler_common
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

  /**
   * Number of objects in the list.
   */
  virtual int count() const;

  /**
   * Number of Range objects in the list.
   */
  virtual int rangeCount() const;

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
