/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Braced.hpp"
#include "src/common/Conditioned.hpp"
#include "src/common/Scoped.hpp"

namespace birch {
/**
 * DoWhile loop.
 *
 * @ingroup statement
 */
class DoWhile: public Statement,
    public Scoped,
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

  /**
   * Destructor.
   */
  virtual ~DoWhile();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
