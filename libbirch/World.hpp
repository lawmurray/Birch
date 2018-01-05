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
class World {
public:
  /**
   * Constructor.
   *
   * @param parent Parent world, if any.
   */
  World(const std::shared_ptr<World>& parent = nullptr);

  /**
   * Retrieve (and possibly create) an allocation.
   *
   * @param src The source allocation.
   *
   * @return The allocation for this import. If no such allocation exists yet,
   * one is created and returned.
   */
  Any* get(Any* src);

  /**
   * Insert a mapping.
   *
   * @param src The source allocation.
   * @param dst The destination allocation.
   */
  void insert(Any* src, Any* dst);

private:
  /**
   * Parent world.
   */
  std::shared_ptr<World> parent;

  /**
   * Mapped allocations.
   */
  std::unordered_map<Any*,Any*> map;
};
}
