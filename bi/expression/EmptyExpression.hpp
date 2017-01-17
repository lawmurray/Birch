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

  virtual possibly dispatch(Expression& o);
  virtual possibly le(EmptyExpression& o);
};
}
