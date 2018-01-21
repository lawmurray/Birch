/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Parameterised statement.
 *
 * @ingroup common
 */
class Parameterised {
public:
  /**
   * Constructor.
   *
   * @param params Parameters.
   */
  Parameterised(Expression* params);

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
