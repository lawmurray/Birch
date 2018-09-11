/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

bi::World::World() :
    cloneSource(nullptr),
    launchSource(fiberWorld),
    launchDepth(fiberWorld->launchDepth + 1) {
  //
}

bi::World::World(int) :
    cloneSource(nullptr),
    launchSource(nullptr),
    launchDepth(0) {
  //
}

bi::World::World(const SharedPtr<World>& cloneSource) :
    cloneSource(cloneSource),
    launchSource(fiberWorld),
    launchDepth(cloneSource->launchDepth) {
  //
}

bi::World::~World() {
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
  return hasCloneAncestor(world)
      || (launchSource && launchSource->hasLaunchAncestor(world));
}

int bi::World::depth() const {
  return launchDepth;
}

bi::Any* bi::World::get(Any* o, World* current) {
  assert(o);
  int d = depth() - current->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(current));
  return pull(o, current, dst);
}

bi::Any* bi::World::getNoCopy(Any* o, World* current) {
  assert(o);
  int d = depth() - current->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(current));
  return pullNoCopy(o, current, dst);
}

bi::Any* bi::pull(Any* o, World* current, World* world) {
  assert(o && world->hasCloneAncestor(current));

  Any* result = pullNoCopy(o, current, world);
  if (world != result->getWorld()) {
    Enter enter(world);
    Clone clone;
    SharedPtr<Any> copied = result->clone();
    result = world->map.set(result, copied.get());
    ///@todo Race condition here if multiple threads running same fiber, may
    ///end up with multiple copies of the one object
  }
  return result;
}

bi::Any* bi::pullNoCopy(Any* o, World* current, World* world) {
  assert(o && world->hasCloneAncestor(current));

  Any *mapped, *result;
  bool fromCache;
  if (world == current) {
    mapped = o;
    fromCache = false;
  } else {
    mapped = world->map.get(o);
    if (!mapped) {
      mapped = pullNoCopy(o, current, world->cloneSource.get());
      world->map.set(o, mapped);
    }
    fromCache = true;
  }

  if (world == mapped->getWorld() || (fromCache && o == mapped)) {
    // ^ second condition is to save a second unnecessary lookup of the same
    //   key in the cache
    result = mapped;
  } else {
    result = world->map.get(mapped);
    if (!result) {
      result = mapped;
    }
  }

  return result;
}
