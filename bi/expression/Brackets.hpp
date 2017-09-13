/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Expression in brackets.
 *
 * @ingroup compiler_expression
 */
class Brackets: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in brackets.
   * @param loc Location.
   */
  Brackets(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Brackets();

  virtual Iterator<Expression> begin() const;
  virtual Iterator<Expression> end() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
