/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Keyword `this`.
 *
 * @ingroup expression
 */
class This: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  This(Location* loc = nullptr);

  virtual bool isThis() const;

  virtual void accept(Visitor* visitor) const;
};
}
