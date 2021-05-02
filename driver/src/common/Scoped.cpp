/**
 * @file
 */
#include "src/common/Scoped.hpp"

birch::Scoped::Scoped(const ScopeCategory category) :
    scope(new Scope(category)) {
  //
}
