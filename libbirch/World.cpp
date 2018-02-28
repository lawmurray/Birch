/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

#include <cassert>
#include <iostream>

bi::World::World() :
    cloneSource(nullptr),
    launchSource(fiberWorld),
    launchDepth(fiberWorld ? fiberWorld->launchDepth + 1 : 0) {
  //
}

bi::World::World(const std::shared_ptr<World>& cloneSource) :
    cloneSource(cloneSource),
    launchSource(fiberWorld),
    launchDepth(cloneSource->launchDepth) {
  //
}

bool bi::World::hasCloneAncestor(const std::weak_ptr<World>& world) const {
  ///@todo Can use weak_from_this() under C++17
  return this == world.lock().get()
      || (cloneSource && cloneSource->hasCloneAncestor(world));
}

bool bi::World::hasLaunchAncestor(const std::weak_ptr<World>& world) const {
  ///@todo Can use weak_from_this() under C++17
  return this == world.lock().get()
      || (launchSource && launchSource->hasLaunchAncestor(world));
}

int bi::World::depth() const {
  return launchDepth;
}

std::shared_ptr<bi::Any> bi::World::get(const std::shared_ptr<Any>& o) {
  assert(o);
  int d = depth() - o->getWorld().lock()->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource.get();
  }
  assert(dst && dst->hasCloneAncestor(o->getWorld()));
  return dst->pullAndCopy(o);
}

std::shared_ptr<bi::Any> bi::World::pullAndCopy(
    const std::shared_ptr<Any>& o) {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld().lock();
  if (this == src.get()) {
    return o;
  } else {
    assert(cloneSource);
    auto result = cloneSource->pull(o);
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
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld().lock();
  if (this == src.get()) {
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
