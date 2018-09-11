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
    result = clone(result, world);
  }
  return result;
}

bi::Any* bi::pullNoCopy(Any* o, World* current, World* world) {
  assert(o && world->hasCloneAncestor(current));

  Any *mapped, *result;
  if (world == current) {
    mapped = o;
  } else {
    mapped = world->map.get(o);
    if (!mapped) {
      mapped = pullNoCopy(o, current, world->cloneSource.get());
      world->map.set(o, mapped);
    }
  }

  if (world == mapped->getWorld()) {
    result = mapped;
  } else {
    result = world->map.get(mapped);
    if (!result) {
      result = mapped;
    }
  }

  return result;
}

bi::Any* bi::clone(Any* o, World* world) {
  Enter enter(world);
  Clone clone;
  SharedPtr<Any> result = o->clone();
  return world->map.set(o, result.get());
  ///@todo Race condition here if multiple threads running same fiber, may
  ///end up with multiple copies of the one object
}
