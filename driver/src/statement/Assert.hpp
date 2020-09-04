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

  /**
   * Destructor.
   */
  virtual ~Assert();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
