/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

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
  const std::shared_ptr<Any>& get(const std::shared_ptr<Any>& o);

  /**
   * Get an object.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  const std::shared_ptr<Any>& getNoCopy(const std::shared_ptr<Any>& o);

private:
  /**
   * Pull and copy (if necessary) an object from a clone ancestor into this
   * world.
   *
   * @param o The object.
   *
   * @return The mapped and copied object.
   */
  const std::shared_ptr<Any>& pull(const std::shared_ptr<Any>& o);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  const std::shared_ptr<Any>& pullNoCopy(const std::shared_ptr<Any>& o) const;

  /**
   * Insert a mapping.
   *
   * @param src The source object.
   * @param dst The destination object.
   *
   * @return Reference to the newly inserted destination object.
   */
  const std::shared_ptr<Any>& insert(const std::shared_ptr<Any>& src,
      const std::shared_ptr<Any>& dst);

  /**
   * The world from which this world was cloned.
   */
  std::shared_ptr<World> cloneSource;

  /**
   * The world from which this world was launched.
   */
  World* launchSource;

  /**
   * Mapped allocations.
   */
  std::map<Any*,std::shared_ptr<Any>> map;

  /**
   * Launch depth.
   */
  int launchDepth;
};
}

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

#include <cassert>

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

inline const std::shared_ptr<bi::Any>& bi::World::get(
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

inline const std::shared_ptr<bi::Any>& bi::World::getNoCopy(
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

inline const std::shared_ptr<bi::Any>& bi::World::pull(
    const std::shared_ptr<Any>& o) {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld();
  if (this == src) {
    return o;
  } else {
    assert(cloneSource);
    auto& result = cloneSource->pullNoCopy(o);
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      Enter enter(this);
      Clone clone;
      return insert(result, result->clone());
    }
  }
}

inline const std::shared_ptr<bi::Any>& bi::World::pullNoCopy(
    const std::shared_ptr<Any>& o) const {
  assert(o && hasCloneAncestor(o->getWorld()));

  auto src = o->getWorld();
  if (this == src) {
    return o;
  } else {
    assert(cloneSource);
    auto& result = cloneSource->pullNoCopy(o);
    auto iter = map.find(result.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      return result;
    }
  }
}

inline const std::shared_ptr<bi::Any>& bi::World::insert(const std::shared_ptr<Any>& src,
    const std::shared_ptr<Any>& dst) {
  auto result = map.insert(std::make_pair(src.get(), dst));
  assert(result.second);
  return result.first->second;
}
