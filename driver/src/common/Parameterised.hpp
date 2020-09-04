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
   * Destructor.
   */
  virtual ~Parameterised() = 0;

  /**
   * Parameters.
   */
  Expression* params;
};
}
