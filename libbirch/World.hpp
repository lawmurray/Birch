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
   * @param parent World from which this was cloned, if any.
   */
  World(const std::shared_ptr<World>& parent = nullptr);

  /**
   * Map an object from its current world into this world, cloning it into
   * this world if necessary.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  std::shared_ptr<Any> get(const std::shared_ptr<Any>& o);

private:
  /**
   * Map an object from its current world into this world.
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
   * The world from which this world was cloned.
   */
  std::shared_ptr<World> parent;

  /**
   * Mapped allocations.
   */
  std::unordered_map<Any*,std::shared_ptr<Any>> map;
};
}
