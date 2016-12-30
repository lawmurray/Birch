/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Bracketed expression.
 *
 * @ingroup compiler_expression
 */
class BracketsExpression: public Expression, public Bracketed {
public:
  /**
   * Constructor.
   *
   * @param expr Expression.
   * @param brackets Brackets.
   * @param loc Location.
   */
  BracketsExpression(Expression* expr, Expression* brackets,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BracketsExpression();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;

  /**
   * Left operand.
   */
  unique_ptr<Expression> expr;
};
}
