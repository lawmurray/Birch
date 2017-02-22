/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Empty expression.
 *
 * @ingroup compiler_expression
 */
class EmptyExpression: public Expression {
public:
  /**
   * Constructor.
   */
  EmptyExpression();

  /**
   * Destructor.
   */
  virtual ~EmptyExpression();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(EmptyExpression& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(EmptyExpression& o);
};
}
