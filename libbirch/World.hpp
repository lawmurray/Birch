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
   * Get the mapped version of the given object for this world.
   *
   * @param src The source object.
   *
   * @return The destination object.
   */
  std::shared_ptr<Any> get(const std::shared_ptr<Any>& src);

private:
  /**
   * Recursively pull an object from its current world this world,
   * applying any mappings along the way.
   *
   * @param src The source object.
   *
   * @return The destination object.
   */
  std::shared_ptr<Any> pull(const std::shared_ptr<Any>& src) const;

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
