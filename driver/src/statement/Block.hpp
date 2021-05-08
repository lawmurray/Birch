/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Braced.hpp"
#include "src/common/Conditioned.hpp"

namespace birch {
/**
 * Block without control flow, but defining a scope.
 *
 * @ingroup statement
 */
class Block: public Statement,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param braces Body of block.
   * @param loc Location.
   */
  Block(Statement* braces, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
