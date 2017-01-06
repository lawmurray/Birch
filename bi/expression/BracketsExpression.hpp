/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/BracketsExpression.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Bracketed expression.
 *
 * @ingroup compiler_expression
 */
class BracketsExpression: public Expression, public ExpressionUnary, public Bracketed {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param brackets Brackets.
   * @param loc Location.
   */
  BracketsExpression(Expression* single, Expression* brackets,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BracketsExpression();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatch(Expression& o);
  virtual bool le(BracketsExpression& o);
  virtual bool le(VarParameter& o);
};
}
