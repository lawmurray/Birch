/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"

#include <cassert>

bi::World::World(const std::shared_ptr<World>& parent) :
    parent(parent) {
  //
}

std::shared_ptr<bi::Any> bi::World::pull(const std::shared_ptr<Any>& src) {
  assert(src);
  if (shared_from_this() == src->getWorld().lock()) {
    return src;
  } else {
    assert(parent);
    auto dst = parent->pull(src);
    auto iter = map.find(dst);
    if (iter != map.end()) {
      dst = iter->second;
    }
    return dst;
  }
}

void bi::World::insert(const std::shared_ptr<Any>& src,
    const std::shared_ptr<Any>& dst) {
  auto result = map.insert(std::make_pair(src, dst));
  assert(result.second);
}
