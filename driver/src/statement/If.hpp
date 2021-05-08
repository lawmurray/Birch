/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Conditioned.hpp"
#include "src/expression/EmptyExpression.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * If.
 *
 * @ingroup statement
 */
class If: public Statement,
    public Conditioned,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param braces True branch.
   * @param falseBraces False branch.
   * @param loc Location.
   */
  If(Expression* cond, Statement* braces, Statement* falseBraces,
      Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;

  /**
   * Block if condition is false. May be empty if there is no else clause.
   */
  Statement* falseBraces;
};
}
