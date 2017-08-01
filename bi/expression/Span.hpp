/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"

namespace bi {
/**
 * Span expression.
 *
 * @ingroup compiler_expression
 */
class Span: public Expression, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Span(Expression* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Span();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
