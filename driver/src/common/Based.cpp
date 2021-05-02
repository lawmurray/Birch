/**
 * @file
 */
#include "src/common/Based.hpp"

birch::Based::Based(Type* base, const bool alias) :
    base(base), alias(alias) {
  //
}

bool birch::Based::isAlias() const {
  return alias;
}
