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
   * @param parent Parent world, if any.
   */
  World(const std::shared_ptr<World>& parent = nullptr);

  /**
   * Is the given world reachable from this world through the ancestry?
   */
  bool isReachable(const World* o) const;

  /**
   * Pull an object from its current world to this world, copying it into
   * the current world if it is not already there.
   *
   * @param src The source object.
   *
   * @return The destination object.
   */
  std::shared_ptr<Any> get(const std::shared_ptr<Any>& src);

  /**
   * Pull an object from its current world to this world,  but without copying
   * it into the current world if it is not already there.
   *
   * @param src The source object.
   *
   * @return The destination object.
   */
  std::shared_ptr<Any> pull(const std::shared_ptr<Any>& src) const;

private:
  /**
   * Insert a mapping.
   *
   * @param src The source object.
   * @param dst The destination object.
   */
  void insert(const std::shared_ptr<Any>& src,
      const std::shared_ptr<Any>& dst);

  /**
   * Parent world.
   */
  std::shared_ptr<World> parent;

  /**
   * Mapped allocations.
   */
  std::unordered_map<Any*,std::shared_ptr<Any>> map;
};
}
