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
  return this == world
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
  assert(o && hasCloneAncestor(current));

  /* map */
  Any* mapped;
  if (world != current) {
    mapped = world->map.get(o);
    if (!mapped) {
      mapped = pullNoCopy(o, current, world->cloneSource.get());
      mapped = world->map.put(o, mapped);
    }
  } else {
    mapped = o;
  }

  /* copy */
  Any* result;
  if (world != mapped->getWorld()) {
    Enter enter(world);
    Clone clone;
    SharedPtr<Any> copied = mapped->clone();
    result = world->map.set(mapped, copied.get());
  } else {
    result = mapped;
  }
  return result;
}

bi::Any* bi::pullNoCopy(Any* o, World* current, World* world) {
  assert(o && hasCloneAncestor(current));

  /* map */
  Any* mapped;
  if (world != current) {
    mapped = world->map.get(o);
    if (!mapped) {
      mapped = pullNoCopy(o, current, world->cloneSource.get());
      mapped = world->map.put(o, mapped);
    }
  } else {
    mapped = o;
  }

  /* previous copy */
  Any* result;
  if (world != mapped->getWorld() && o != mapped) {
    result = world->map.get(mapped);
    if (!result) {
      result = mapped;
    }
  } else {
    result = mapped;
  }
  return result;
}
