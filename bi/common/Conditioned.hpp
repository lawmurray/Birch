/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"

namespace bi {
/**
 * Statement with a condition (e.g. conditional, loop).
 *
 * @ingroup birch_common
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
