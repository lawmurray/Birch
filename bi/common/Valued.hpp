/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Statement or expression with a default or initial value.
 *
 * @ingroup birch_common
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
