/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/expression/EmptyExpression.hpp"

namespace birch {
/**
 * Expression with arguments.
 *
 * @ingroup common
 */
class Argumented {
public:
  /**
   * Constructor.
   *
   * @param args Arguments.
   */
  Argumented(Expression* args);

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
