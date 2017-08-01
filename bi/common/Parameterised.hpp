/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
/**
 * Parameterised expression.
 *
 * @ingroup compiler_common
 */
class Parameterised {
public:
  /**
   * Constructor.
   *
   * @param params Parameters.
   */
  Parameterised(Expression* params = new EmptyExpression());

  /**
   * Destructor.
   */
  virtual ~Parameterised() = 0;

  /**
   * Parameters.
   */
  Expression* params;
};
}
