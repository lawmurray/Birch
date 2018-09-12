/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

bi::World::World() :
    cloneSource(nullptr) {
  //
}

bi::World::World(int) :
    cloneSource(nullptr) {
  incShared();
}

bi::World::World(const SharedPtr<World>& cloneSource) :
    cloneSource(cloneSource) {
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

bi::Any* bi::World::get(Any* o, World* current) {
  assert(o);
  return pull(o, current, this);
}

bi::Any* bi::World::getNoCopy(Any* o, World* current) {
  assert(o);
  return pullNoCopy(o, current, this);
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
