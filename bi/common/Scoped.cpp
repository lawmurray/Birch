/**
 * @file
 */
#include "bi/common/Scoped.hpp"

bi::Scoped::Scoped(Scope* scope) :
    scope(scope) {
  //
}

bi::Scoped::Scoped(const ScopeCategory category) :
    scope(new Scope(category)) {
  //
}

bi::Scoped::~Scoped() {
  //
}
