/**
 * @file
 */
#include "src/common/Scoped.hpp"

birch::Scoped::Scoped(Scope* scope) :
    scope(scope) {
  //
}

birch::Scoped::Scoped(const ScopeCategory category) :
    scope(new Scope(category)) {
  //
}
