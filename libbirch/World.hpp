/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

#include <unordered_map>

namespace bi {
/**
 * Fiber world.
 *
 * @ingroup libbirch
 */
class World: public std::enable_shared_from_this<World> {
public:
  /**
   * Constructor.
   *
   * @param launchSource World from which this was launched.
   * @param cloneSource World from which this was cloned.
   */
  World(const std::shared_ptr<World>& launchSource = nullptr,
      const std::shared_ptr<World>& cloneSource = nullptr);

  /**
   * Does this world have the given world as a launch ancestor?
   */
  bool hasLaunchAncestor(const std::shared_ptr<World>& o) const;

  /**
   * Does this world have the given world as a clone ancestor?
   */
  bool hasCloneAncestor(const std::shared_ptr<World>& o) const;

  /**
   * Map an object to the present context.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  std::shared_ptr<Any> get(const std::shared_ptr<Any>& o);

private:
  /**
   * Pull an object from a clone ancestor into this world, creating a copy
   * of its latest incarnation in this world if it doesn't already exist.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  std::shared_ptr<Any> pullAndCopy(const std::shared_ptr<Any>& o);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  std::shared_ptr<Any> pull(const std::shared_ptr<Any>& o) const;

  /**
   * Insert a mapping.
   *
   * @param src The source object.
   * @param dst The destination object.
   */
  void insert(const std::shared_ptr<Any>& src,
      const std::shared_ptr<Any>& dst);

  /**
   * The world from which this world was launched.
   */
  std::shared_ptr<World> launchSource;

  /**
   * The world from which this world was cloned.
   */
  std::shared_ptr<World> cloneSource;

  /**
   * Mapped allocations.
   */
  std::unordered_map<Any*,std::shared_ptr<Any>> map;
};
}
