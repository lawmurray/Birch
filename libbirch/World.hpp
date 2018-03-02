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
   * Map an object to the present context.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  const std::shared_ptr<Any>& get(const std::shared_ptr<Any>& o);

private:
  /**
   * Pull an object from a clone ancestor into this world, creating a copy
   * of its latest incarnation in this world if it doesn't already exist.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  const std::shared_ptr<Any>& pullAndCopy(const std::shared_ptr<Any>& o);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  const std::shared_ptr<Any>& pull(const std::shared_ptr<Any>& o) const;

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
