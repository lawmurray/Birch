/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Conditioned.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * With loop.
 *
 * @ingroup statement
 */
class With: public Statement,
    public Single<Expression>,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param single Expression giving handler.
   * @param braces Body of loop.
   * @param loc Location.
   */
  With(Expression* single, Statement* braces, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
