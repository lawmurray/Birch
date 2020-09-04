/**
 * @file
 */
#pragma once

#include "src/common/Scope.hpp"

namespace birch {
/**
 * Statement with a scope.
 *
 * @ingroup common
 */
class Scoped {
public:
  /**
   * Constructor.
   */
  Scoped(Scope* scope);

  /**
   * Constructor.
   */
  Scoped(const ScopeCategory category);

  /**
   * Destructor.
   */
  virtual ~Scoped() = 0;

  /**
   * Scope.
   */
  Scope* scope;
};
}
