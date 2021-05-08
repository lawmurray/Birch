/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * ExpressionStatement.
 *
 * @ingroup statement
 */
class ExpressionStatement: public Statement, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  ExpressionStatement(Expression* single, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
