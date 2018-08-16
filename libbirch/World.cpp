/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

bi::World::World() :
    launchSource(fiberWorld),
    launchDepth(fiberWorld->launchDepth + 1) {
  //
}

bi::World::World(int) :
    launchDepth(0) {
  //
}

bi::World::World(const SharedPtr<World>& cloneSource) :
    cloneSource(cloneSource),
    launchSource(fiberWorld),
    launchDepth(cloneSource->launchDepth) {
  //
}

void bi::World::destroy() {
  size = sizeof(*this);
  this->~World();
}

bool bi::World::hasCloneAncestor(World* world) const {
  return this == world
      || (cloneSource && cloneSource->hasCloneAncestor(world));
}

bool bi::World::hasLaunchAncestor(World* world) const {
  return this == world
      || (launchSource && launchSource->hasLaunchAncestor(world));
}

int bi::World::depth() const {
  return launchDepth;
}

bi::SharedPtr<bi::Any> bi::World::get(const SharedPtr<Any>& o, World* world) {
  assert(o);
  int d = depth() - world->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(world));
  return dst->pull(o, world);
}

bi::SharedPtr<bi::Any> bi::World::getNoCopy(const SharedPtr<Any>& o,
    World* world) {
  assert(o);
  int d = depth() - world->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(world));
  return dst->pullNoCopy(o, world);
}

bi::SharedPtr<bi::Any> bi::World::pull(const SharedPtr<Any>& o,
    World* world) {
  assert(o && hasCloneAncestor(world));

  SharedPtr<bi::Any> result;
  auto src = world;
  if (this == src) {
    result = o;
  } else {
    /* through cache */
    set();
    auto iter = cache.find(o.get());
    if (iter != cache.end()) {
      result = iter->second;
    } else {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o, world);
      auto ret = cache.insert(std::make_pair(o.get(), result.get()));
      assert(ret.second);
    }
    unset();
  }

  /* through map */
  if (this != result->getWorld()) {
    set();
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      result = iter->second;
    } else {
      Enter enter(this);
      Clone clone;
      auto src = result.get();
      result = result->clone();
      auto ret = map.insert(std::make_pair(src, result));
      assert(ret.second);
    }
    unset();
  }

  return result;
}

bi::SharedPtr<bi::Any> bi::World::pullNoCopy(const SharedPtr<Any>& o,
    World* world) {
  assert(o && hasCloneAncestor(world));

  SharedPtr<bi::Any> result;
  auto src = world;
  if (this == src) {
    result = o;
  } else {
    /* through cache */
    set();
    auto iter = cache.find(o.get());
    if (iter != cache.end()) {
      result = iter->second;
    } else {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o, world);
      auto ret = cache.insert(std::make_pair(o.get(), result.get()));
      assert(ret.second);
    }
    unset();
  }

  /* through map */
  if (this != result->getWorld()) {
    set();
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      result = iter->second;
    }
    unset();
  }

  return result;
}
