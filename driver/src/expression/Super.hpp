/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Keyword `super`.
 *
 * @ingroup expression
 */
class Super: public Expression {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Super(Location* loc = nullptr);

  virtual bool isSuper() const;

  virtual void accept(Visitor* visitor) const;
};
}
