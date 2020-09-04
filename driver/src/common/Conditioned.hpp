/**
 * @file
 */
#pragma once

#include "src/expression/all.hpp"

namespace birch {
/**
 * Statement with a condition (e.g. conditional, loop).
 *
 * @ingroup common
 */
class Conditioned {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   */
  Conditioned(Expression* cond);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

  /**
   * Condition.
   */
  Expression* cond;
};
}
