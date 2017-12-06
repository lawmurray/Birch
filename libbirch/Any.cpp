/**
 * @file
 */
#include "libbirch/Any.hpp"

bi::Any::Any() {
  //
}

bi::Any::Any(const world_t world, const Any& o) :
    ptr() {
  //
}

bi::Any::~Any() {
  //
}

bi::Any* bi::Any::clone(const world_t world) {
  return new Any(world, *this);
}
