/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/PowerPoolAllocator.hpp"

#include <map>

namespace bi {
/**
 * Fiber world.
 *
 * @ingroup libbirch
 */
class World: public std::enable_shared_from_this<World> {
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
  World(const std::shared_ptr<World>& cloneSource);

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
  std::shared_ptr<Any> get(const std::shared_ptr<Any>& o);

  /**
   * Get an object.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  std::shared_ptr<Any> getNoCopy(const std::shared_ptr<Any>& o);

private:
  /**
   * Pull and copy (if necessary) an object from a clone ancestor into this
   * world.
   *
   * @param o The object.
   *
   * @return The mapped and copied object.
   */
  std::shared_ptr<Any> pull(const std::shared_ptr<Any>& o);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  std::shared_ptr<Any> pullNoCopy(const std::shared_ptr<Any>& o);

  /**
   * The world from which this world was cloned.
   */
  std::shared_ptr<World> cloneSource;

  /**
   * The world from which this world was launched.
   */
  World* launchSource;

  /*
   * Types for maps.
   */
  using key_type = Any*;
  using value_type = std::shared_ptr<Any>;
  using less_type = std::less<key_type>;
  using alloc_type = PowerPoolAllocator<std::pair<const key_type,value_type>>;
  using map_type = std::map<key_type,value_type,less_type,alloc_type>;

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

inline bi::World::World(const std::shared_ptr<World>& cloneSource) :
    cloneSource(cloneSource),
    launchSource(fiberWorld),
    launchDepth(cloneSource->launchDepth) {
  //
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

inline std::shared_ptr<bi::Any> bi::World::get(
    const std::shared_ptr<Any>& o) {
  assert(o);
  int d = depth() - o->getWorld()->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(o->getWorld()));
  return dst->pull(o);
}

inline std::shared_ptr<bi::Any> bi::World::getNoCopy(
    const std::shared_ptr<Any>& o) {
  assert(o);
  int d = depth() - o->getWorld()->depth();
  assert(d >= 0);
  auto dst = this;
  for (int i = 0; i < d; ++i) {
    dst = dst->launchSource;
    assert(dst);
  }
  assert(dst->hasCloneAncestor(o->getWorld()));
  return dst->pullNoCopy(o);
}

inline std::shared_ptr<bi::Any> bi::World::pull(
    const std::shared_ptr<Any>& o) {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld();
  if (this == src) {
    return o;
  } else {
    std::shared_ptr<bi::Any> result;

    /* through cache */
    auto iter = cache.find(o.get());
    if (iter != cache.end()) {
      result = iter->second;
    } else {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o);
      auto ret = cache.insert(std::make_pair(o.get(), result));
      assert(ret.second);
    }

    /* through map */
    iter = map.find(result.get());
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

inline std::shared_ptr<bi::Any> bi::World::pullNoCopy(
    const std::shared_ptr<Any>& o) {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld();
  if (this == src) {
    return o;
  } else {
    std::shared_ptr<bi::Any> result;

    /* check cache */
    auto iter = cache.find(o.get());
    if (iter != cache.end()) {
      result = iter->second;
    } else {
      assert(cloneSource);
      result = cloneSource->pullNoCopy(o);
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
