/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/Allocator.hpp"

#include <unordered_map>

namespace bi {
/**
 * Fiber world.
 *
 * @ingroup libbirch
 */
class World: public Counted {
public:
  /**
   * Default constructor.
   */
  World();

  /**
   * Constructor for root.
   */
  World(int);

  /**
   * Constructor for clone.
   *
   * @param cloneSource Clone parent.
   */
  World(const SharedPtr<World>& cloneSource);

  /**
   * Deallocate.
   */
  virtual void destroy();

  /**
   * Does this world have the given world as a clone ancestor?
   */
  bool hasCloneAncestor(World* world) const;

  /**
   * Does this world have the given world as a launch ancestor?
   */
  bool hasLaunchAncestor(World* world) const;

  /**
   * Get launch depth.
   */
  int depth() const;

  /**
   * Get an object, copying it if necessary.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  SharedPtr<Any> get(const SharedPtr<Any>& o, World* world);

  /**
   * Get an object.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  SharedPtr<Any> getNoCopy(const SharedPtr<Any>& o, World* world);

private:
  /**
   * Pull and copy (if necessary) an object from a clone ancestor into this
   * world.
   *
   * @param o The object.
   *
   * @return The mapped and copied object.
   */
  SharedPtr<Any> pull(const SharedPtr<Any>& o, World* world);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  SharedPtr<Any> pullNoCopy(const SharedPtr<Any>& o,
      World* world);

  /**
   * The world from which this world was cloned.
   */
  SharedPtr<World> cloneSource;

  /**
   * The world from which this world was launched.
   */
  World* launchSource;

  /*
   * Types for maps.
   */
  using key_type = Any*;
  using value_type = SharedPtr<Any>;
  using hash_type = std::hash<key_type>;
  using equal_type = std::equal_to<key_type>;
  using alloc_type = Allocator<std::pair<const key_type,value_type>>;
  using map_type = std::unordered_map<key_type,value_type,hash_type,equal_type,alloc_type>;

  /**
   * Mapped allocations.
   */
  map_type map;

  /**
   * Cached mappings of clone ancestors.
   */
  map_type cache;

  /**
   * Launch depth.
   */
  int launchDepth;
};
}

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

inline bi::World::World() :
    launchSource(fiberWorld),
    launchDepth(fiberWorld->launchDepth + 1) {
  //
}

inline bi::World::World(int) :
    launchDepth(0) {
  //
}

inline bi::World::World(const SharedPtr<World>& cloneSource) :
    cloneSource(cloneSource),
    launchSource(fiberWorld),
    launchDepth(cloneSource->launchDepth) {
  //
}

inline void bi::World::destroy() {
  ptr = this;
  size = sizeof(*this);
  this->~World();
}

inline bool bi::World::hasCloneAncestor(World* world) const {
  return this == world
      || (cloneSource && cloneSource->hasCloneAncestor(world));
}

inline bool bi::World::hasLaunchAncestor(World* world) const {
  return this == world
      || (launchSource && launchSource->hasLaunchAncestor(world));
}

inline int bi::World::depth() const {
  return launchDepth;
}

inline bi::SharedPtr<bi::Any> bi::World::get(const SharedPtr<Any>& o,
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
  return dst->pull(o, world);
}

inline bi::SharedPtr<bi::Any> bi::World::getNoCopy(
    const SharedPtr<Any>& o, World* world) {
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

inline bi::SharedPtr<bi::Any> bi::World::pull(const SharedPtr<Any>& o,
    World* world) {
  assert(o && hasCloneAncestor(world));

  SharedPtr<bi::Any> result;
  auto src = world;
  if (this == src) {
    result = o;
  } else {
    /* through cache */
    auto iter = cache.find(o.get());
    if (iter != cache.end()) {
      result = iter->second;
    } else {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o, world);
      auto ret = cache.insert(std::make_pair(o.get(), result));
      assert(ret.second);
    }
  }

  /* through map */
  if (this == result->getWorld()) {
    return result;
  } else {
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      Enter enter(this);
      Clone clone;
      auto src = result.get();
      result = result->clone();
      auto ret = map.insert(std::make_pair(src, result));
      assert(ret.second);
      return result;
    }
  }
}

inline bi::SharedPtr<bi::Any> bi::World::pullNoCopy(
    const SharedPtr<Any>& o, World* world) {
  assert(o && hasCloneAncestor(world));

  auto src = world;
  if (this == src) {
    return o;
  } else {
    SharedPtr<bi::Any> result;

    /* check cache */
    auto iter = cache.find(o.get());
    if (iter != cache.end()) {
      result = iter->second;
    } else {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o, world);
      auto ret = cache.insert(std::make_pair(o.get(), result));
      assert(ret.second);
    }

    /* map through copies */
    iter = map.find(result.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      return result;
    }
  }
}
