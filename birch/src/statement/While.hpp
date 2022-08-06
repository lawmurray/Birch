/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Conditioned.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * While loop.
 *
 * @ingroup statement
 */
class While: public Statement,
    public Conditioned,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param braces Body of loop.
   * @param loc Location.
   */
  While(Expression* cond, Statement* braces, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
