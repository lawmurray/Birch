/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Span expression.
 *
 * @ingroup birch_expression
 */
class Span: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Span(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Span();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
