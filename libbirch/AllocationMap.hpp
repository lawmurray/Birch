/**
 * @file
 */
#pragma once

#include <unordered_map>

namespace bi {
class Allocation;
class Any;

/**
 * Allocation map for fiber.
 *
 * @ingroup libbirch
 */
class AllocationMap {
public:
  /**
   * Get an allocation through the mapping.
   *
   * @param from The original .
   *
   * @return The new pointer, or the original pointer if no mapping is
   * provided for it.
   */
  Any* get(Allocation* from) const;

  /**
   * Set a mapping.
   *
   * @param from The original pointer.
   * @param to The new pointer.
   */
  void set(Allocation* from, Any* to);

private:
  /**
   * Mapped allocations.
   */
  std::unordered_map<Allocation*,Any*> map;
};
}
