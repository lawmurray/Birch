/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Clone.hpp"

#include <cassert>

bi::World::World(const std::shared_ptr<World>& parent) :
    parent(parent) {
  //
}

std::shared_ptr<bi::Any> bi::World::get(const std::shared_ptr<Any>& o) {
  auto src = o->getWorld().get();
  if (this == src) {
    return o;
  } else {
    assert(parent);
    auto result = parent->pull(o);
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      Enter enter(shared_from_this());
      Clone clone;
      auto copy = result->clone();
      insert(result, copy);
      return copy;
    }
  }
}

std::shared_ptr<bi::Any> bi::World::pull(
    const std::shared_ptr<Any>& o) const {
  auto src = o->getWorld().get();
  if (this == src) {
    return o;
  } else {
    assert(parent);
    auto result = parent->pull(o);
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      result = iter->second;
    }
    return result;
  }
}

void bi::World::insert(const std::shared_ptr<Any>& src,
    const std::shared_ptr<Any>& dst) {
  auto result = map.insert(std::make_pair(src.get(), dst));
  assert(result.second);
}
