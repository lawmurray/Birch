/**
 * @file
 */
#pragma once

#include "src/common/Name.hpp"
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
   * @param op Initialization operator.
   * @param value Value.
   */
  Valued(Name* op, Expression* value);

  /**
   * Initialization operator.
   */
  Name* op;

  /**
   * Initial value.
   */
  Expression* value;
};
}
