/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Expression with proceeding square brackets.
 *
 * @ingroup compiler_expression
 */
class BracketsExpression: public Expression,
    public Unary<Expression>,
    public Bracketed {
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

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const BracketsExpression& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const BracketsExpression& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
