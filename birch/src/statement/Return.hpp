/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Return statement.
 *
 * @ingroup statement
 */
class Return: public Statement, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Return(Expression* single, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
