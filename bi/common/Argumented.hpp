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

  /**
   * After resolution, the type of the function called.
   */
  FunctionType* callType;
};
}
