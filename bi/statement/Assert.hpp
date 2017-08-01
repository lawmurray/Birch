/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Conditioned.hpp"

namespace bi {
/**
 * Assertion statement.
 *
 * @ingroup compiler_statement
 */
class Assert: public Statement, public Conditioned {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param loc Location.
   */
  Assert(Expression* cond, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Assert();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
