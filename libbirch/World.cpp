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
  map.decShared();
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

  Any* result = nullptr;
  auto src = world;
  size_t i;
  bool inserted;
  if (this == src) {
    result = o;
  } else {
    /* through cache */
    std::tie(i, inserted) = cache.claim(o);
    if (inserted) {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o, world);
      cache.write(i, result);
    } else {
      result = cache.read(i);
    }
  }

  /* through map */
  if (this != result->getWorld()) {
    std::tie(i, inserted) = map.claim(result);
    if (inserted) {
      Enter enter(this);
      Clone clone;
      result = result->clone();
      result->incShared();
      map.write(i, result);
    } else {
      result = map.read(i);
    }
  }

  return result;
}

bi::Any* bi::World::pullNoCopy(Any* o, World* world) {
  assert(o && hasCloneAncestor(world));

  Any* result = nullptr;
  auto src = world;
  if (this == src) {
    result = o;
  } else {
    /* through cache */
    size_t i;
    bool inserted;
    std::tie(i, inserted) = cache.claim(o);
    if (inserted) {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o, world);
      cache.write(i, result);
    } else {
      result = cache.read(i);
    }
  }

  /* through map */
  if (this != result->getWorld()) {
    auto to = map.get(result);
    if (to) {
      result = to;
    }
  }

  return result;
}
