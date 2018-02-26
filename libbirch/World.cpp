/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

#include <cassert>

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

bool bi::World::hasCloneAncestor(const World* world) const {
  return this == world
      || (cloneSource && cloneSource->hasCloneAncestor(world));
}

bool bi::World::hasLaunchAncestor(const World* world) const {
  return this == world
      || (launchSource && launchSource->hasLaunchAncestor(world));
}

int bi::World::depth() const {
  return launchDepth;
}

std::shared_ptr<bi::Any> bi::World::get(const std::shared_ptr<Any>& o) {
  assert(o);
  int d = depth() - o->getWorld()->depth();
  if (d >= 0) {
    auto dst = this;
    for (int i = 0; i < d && dst; ++i) {
      dst = dst->launchSource.get();
    }
    assert(dst && dst->hasCloneAncestor(o->getWorld().get()));
    return dst->pullAndCopy(o);
  } else {
    return o;
  }
}

std::shared_ptr<bi::Any> bi::World::pullAndCopy(
    const std::shared_ptr<Any>& o) {
  assert(o && hasCloneAncestor(o->getWorld().get()));

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
  assert(o && hasCloneAncestor(o->getWorld().get()));

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
