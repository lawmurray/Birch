/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

bi::World::World() :
    launchSource(fiberWorld),
    map(256u),
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

bi::Any* bi::World::get(Any* o, World* world) {
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

bi::Any* bi::World::getNoCopy(Any* o, World* world) {
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

bi::Any* bi::World::pull(Any* o, World* world) {
  assert(o && hasCloneAncestor(world));

  /* map */
  auto mapped = (this == world) ? o : map.getOrPut(o, [=]() {
    return cloneSource->pullNoCopy(o, world);
  });

  /* copy */
  if (this != mapped->getWorld()) {
    Enter enter(this);
    Clone clone;
    auto copied = mapped->clone();
    map.setOrPut(mapped, copied);
    return copied;
  } else {
    return mapped;
  }
}

bi::Any* bi::World::pullNoCopy(Any* o, World* world) {
  assert(o && hasCloneAncestor(world));

  auto mapped = (this == world) ? o : map.getOrPut(o, [=]() {
    return cloneSource->pullNoCopy(o, world);
  });
  if (this != mapped->getWorld() && o != mapped) {
    mapped = map.get(mapped, mapped);
  }
  return mapped;
}
