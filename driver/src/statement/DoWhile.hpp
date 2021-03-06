/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Braced.hpp"
#include "src/common/Conditioned.hpp"

namespace birch {
/**
 * DoWhile loop.
 *
 * @ingroup statement
 */
class DoWhile: public Statement,
    public Braced,
    public Conditioned {
public:
  /**
   * Constructor.
   *
   * @param braces Body of loop.
   * @param cond Condition.
   * @param loc Location.
   */
  DoWhile(Statement* braces, Expression* cond, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
