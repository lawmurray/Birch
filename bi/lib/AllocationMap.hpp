/**
 * @file
 */
#pragma once

#include <map>

namespace bi {
class Any;

/**
 * Allocations map for fiber.
 *
 * @ingroup library
 */
class AllocationMap {
public:
  /**
   * Constructor.
   */
  AllocationMap();

  /**
   * Copy constructor.
   */
  AllocationMap(const AllocationMap& o);

  /**
   * Move constructor.
   */
  AllocationMap(AllocationMap&& o) = default;

  /**
   * Copy assignment.
   */
  AllocationMap& operator=(const AllocationMap& o);

  /**
   * Move assignment.
   */
  AllocationMap& operator=(AllocationMap&& o) = default;

  /**
   * Clone the object.
   */
  virtual AllocationMap* clone();

  /**
   * Get an allocation through the mapping.
   *
   * @param from The original pointer.
   *
   * @return The new pointer, or the original pointer if no mapping is
   * provided for it.
   */
  Any* get(Any* from);

  /**
   * Set a mapping.
   *
   * @param from The original pointer.
   * @param to The new pointer.
   */
  void set(Any* from, Any* to);

//private:
  /**
   * Mapped allocations.
   */
  std::map<Any*,Any*> map;

  /**
   * Generation.
   */
  size_t gen;
};
}
