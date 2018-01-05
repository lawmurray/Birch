/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"

#include <cassert>

bi::World::World(const std::shared_ptr<World>& parent) : parent(parent) {
  //
}

bi::Any* bi::World::get(Any* src) {
  Any* dst = src;
  if (parent) {
    dst = parent->get(dst);
  }
  auto iter = map.find(dst);
  if (iter != map.end()) {
    dst = iter->second;
  }
  return dst;
}

void bi::World::insert(Any* src, Any* dst) {
  auto result = map.insert(std::make_pair(src, dst));
  assert(result.second);
}
