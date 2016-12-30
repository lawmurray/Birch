/**
 * @file
 */
#pragma once

#include "bi/common/Scope.hpp"
#include "bi/primitive/shared_ptr.hpp"

namespace bi {
class Scope;

/**
 * Statement with a scope.
 *
 * @ingroup compiler_common
 */
class Scoped {
public:
  /**
   * Destructor.
   */
  virtual ~Scoped() = 0;

  /**
   * Scope.
   */
  shared_ptr<Scope> scope;
};
}
