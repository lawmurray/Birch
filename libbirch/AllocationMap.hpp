/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Pointer.hpp"

#include "gc/gc_allocator.h"

#include <unordered_map>

namespace bi {
/**
 * Allocation map for fiber.
 *
 * @ingroup libbirch
 */
class AllocationMap : public Any {
public:
  using pointer_type = Pointer<Any>;
  using hash_type = std::hash<pointer_type>;
  using equal_to_type = std::equal_to<pointer_type>;

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
  Pointer<Any> get(const pointer_type& from) const;

  /**
   * Set a mapping.
   *
   * @param from The original pointer.
   * @param to The new pointer.
   */
  void set(const pointer_type& from, const pointer_type& to);

private:
  /**
   * Mapped allocations.
   */
  std::unordered_map<pointer_type,pointer_type,hash_type,equal_to_type,
      gc_allocator<std::pair<const pointer_type,pointer_type>>> map;
};
}
