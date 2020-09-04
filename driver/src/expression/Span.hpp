/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Span expression.
 *
 * @ingroup expression
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
