/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Get expression.
 *
 * @ingroup expression
 */
class Get: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Get(Expression* single, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
