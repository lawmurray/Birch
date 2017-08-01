/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Conditioned.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * While loop.
 *
 * @ingroup compiler_statement
 */
class While: public Statement, public Conditioned, public Braced, public Scoped {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param braces Body of loop.
   * @param loc Location.
   */
  While(Expression* cond, Statement* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~While();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
