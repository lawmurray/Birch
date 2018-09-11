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

bi::Any* bi::World::get(Any* o) {
  assert(o);
  int d = depth() - o->getWorld()->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(o->getWorld()));
  return pull(o, dst);
}

bi::Any* bi::World::getNoCopy(Any* o) {
  assert(o);
  int d = depth() - o->getWorld()->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(o->getWorld()));
  return pullNoCopy(o, dst);
}

bi::Any* bi::pull(Any* o, World* world) {
  assert(o && world->hasCloneAncestor(o->getWorld()));


  /* map */
  Any* mapped = pullNoCopy(o, world);

  /* copy */
  Any* result;
  if (world != mapped->getWorld()) {
    Enter enter(world);
    Clone clone;
    SharedPtr<Any> copied = mapped->clone();
    result = world->map.set(mapped, copied.get());
    ///@todo Race condition here if multiple threads running same fiber, may
    ///end up with multiple copies of the one object
  } else {
    result = mapped;
  }
  return result;
}

bi::Any* bi::pullNoCopy(Any* o, World* world) {
  assert(o && world->hasCloneAncestor(o->getWorld()));

  /* map */
  Any* mapped;
  if (world != o->getWorld()) {
    mapped = world->map.get(o);
    if (!mapped) {
      mapped = pullNoCopy(o, world->cloneSource.get());
      mapped = world->map.put(o, mapped);
    }
  } else {
    mapped = o;
  }

  /* previous copy */
  Any* result;
  if (world != mapped->getWorld()) {
    result = world->map.get(mapped);
    if (!result) {
      result = mapped;
    }
  } else {
    result = mapped;
  }

  return result;
}
