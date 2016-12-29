/**
 * @file
 */
#pragma once

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
   * Constructor.
   *
   * @param scope Scope.
   */
  Scoped();

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

#include "bi/common/Scope.hpp"

inline bi::Scoped::Scoped() {
  //
}

inline bi::Scoped::~Scoped() {
  //
}
