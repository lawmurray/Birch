/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"

#include <cassert>

bi::World::World(const std::shared_ptr<World>& cloneSource) :
    cloneSource(cloneSource),
    launchSource(fiberWorld) {
  //
}

bool bi::World::hasCloneAncestor(const std::shared_ptr<World>& world) const {
  return this == world.get() || (cloneSource &&
      cloneSource->hasCloneAncestor(world));
}

bool bi::World::hasLaunchAncestor(const std::shared_ptr<World>& world) const {
  return this == world.get() || (launchSource &&
      launchSource->hasLaunchAncestor(world));
}

std::shared_ptr<bi::Any> bi::World::get(const std::shared_ptr<Any>& o) {
  assert(o);
  auto src = o->getWorld();
  auto dst = this;
  while (dst && !dst->hasCloneAncestor(src)) {
    dst = dst->launchSource.get();
  }
  if (dst) {
    return dst->pullAndCopy(o);
  } else {
    return o;
  }
}

std::shared_ptr<bi::Any> bi::World::pullAndCopy(const std::shared_ptr<Any>& o) {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld().get();
  if (this == src) {
    return o;
  } else {
    assert(cloneSource);
    auto result = cloneSource->pull(o);
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      auto prevWorld = fiberWorld;
      fiberWorld = shared_from_this();
      auto clone = result->clone();
      fiberWorld = prevWorld;
      insert(result, clone);
      return clone;
    }
  }
}

std::shared_ptr<bi::Any> bi::World::pull(
    const std::shared_ptr<Any>& o) const {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld().get();
  if (this == src) {
    return o;
  } else {
    assert(cloneSource);
    auto result = cloneSource->pull(o);
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
