/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/expression/EmptyExpression.hpp"

namespace birch {
/**
 * Statement or expression with a default or initial value.
 *
 * @ingroup common
 */
class Valued {
public:
  /**
   * Constructor.
   *
   * @param value Value.
   */
  Valued(Expression* value);

  /**
   * Destructor.
   */
  virtual ~Valued() = 0;

  /**
   * Condition.
   */
  Expression* value;
};
}
