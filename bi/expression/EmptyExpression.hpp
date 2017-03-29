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

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const EmptyExpression& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const EmptyExpression& o) const;
};
}
