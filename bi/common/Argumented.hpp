/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Expression with arguments.
 *
 * @ingroup compiler_common
 */
class Argumented {
public:
  /**
   * Constructor.
   *
   * @param args Arguments.
   */
  Argumented(Expression* args = new EmptyExpression());

  /**
   * Destructor.
   */
  virtual ~Argumented() = 0;

  /**
   * Arguments.
   */
  Expression* args;
};
}
