/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Conditioned.hpp"

namespace birch {
/**
 * Assertion statement.
 *
 * @ingroup statement
 */
class Assert: public Statement, public Conditioned {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param loc Location.
   */
  Assert(Expression* cond, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
