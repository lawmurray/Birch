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

  Any *mapped, *copied, *result, *cached = nullptr;
  if (world == current) {
    cached = o;
    mapped = o;
  } else {
    cached = world->map.get(o);
    mapped =
        cached ? cached : pullNoCopy(o, current, world->cloneSource.get());
  }

  if (world == mapped->getWorld()) {
    result = mapped;
  } else {
    copied = world->map.get(mapped);
    if (copied) {
      if (world == copied->getWorld()) {
        result = copied;
      } else {
        result = clone(copied, world);
      }
    } else {
      result = clone(mapped, world);
    }
    if (result != cached) {
      result = world->map.set(o, result);
    }
  }

  return result;
}

bi::Any* bi::pullNoCopy(Any* o, World* current, World* world) {
  assert(o && world->hasCloneAncestor(current));

  Any *mapped, *copied, *result, *cached = nullptr;
  if (world == current) {
    cached = o;
    mapped = o;
  } else {
    cached = world->map.get(o);
    mapped =
        cached ? cached : pullNoCopy(o, current, world->cloneSource.get());
  }

  if (world == mapped->getWorld()) {
    result = mapped;
  } else {
    copied = world->map.get(mapped);
    result = copied ? copied : mapped;
    if (result != cached) {
      result = world->map.set(o, result);
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
