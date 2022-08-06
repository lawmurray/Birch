/**
 * @file
 */
#pragma once

namespace birch {
class Expression;

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
   * Parameters.
   */
  Expression* params;
};
}
