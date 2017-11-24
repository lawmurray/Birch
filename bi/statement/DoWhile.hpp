/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Conditioned.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * DoWhile loop.
 *
 * @ingroup birch_statement
 */
class DoWhile: public Statement,
    public Braced,
    public Conditioned,
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param braces Body of loop.
   * @param cond Condition.
   * @param loc Location.
   */
  DoWhile(Statement* braces, Expression* cond, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~DoWhile();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
