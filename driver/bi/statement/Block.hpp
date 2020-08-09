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
 * Block without control flow, but defining a scope.
 *
 * @ingroup statement
 */
class Block: public Statement,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param braces Body of block.
   * @param loc Location.
   */
  Block(Statement* braces, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Block();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
