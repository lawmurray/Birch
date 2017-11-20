/**
 * @file
 */
#pragma once

#include "bi/lib/Pointer.hpp"
#include "bi/lib/Any.hpp"
#include "bi/lib/atomic_allocator.hpp"

#include "gc/gc_allocator.h"

#include <unordered_map>
#include <list>

namespace std {
template<>
struct hash<bi::Pointer<bi::Any>> : public std::hash<bi::Any*> {
  size_t operator()(const bi::Pointer<bi::Any>& o) const {
    /* the generation is ignored in the hash, as it is reasonably unlikely
     * for two pointers with the same raw pointer but different generation to
     * occur in the same allocation map; this only occurs if memory is
     * garbage collected and reused within the same fiber */
    return std::hash<bi::Any*>::operator()(o.raw);
  }
};

template<>
struct equal_to<bi::Pointer<bi::Any>> {
  bool operator()(const bi::Pointer<bi::Any>& o1,
      const bi::Pointer<bi::Any>& o2) const {
    return o1.raw == o2.raw && o1.gen == o2.gen;
  }
};
}

namespace bi {
/**
 * Allocation map for fiber.
 *
 * @ingroup library
 */
class AllocationMap {
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
  Pointer<Any> get(const pointer_type& from);

  /**
   * Set a mapping.
   *
   * @param from The original pointer.
   * @param to The new pointer.
   */
  void set(const pointer_type& from, const pointer_type& to);

//private:
  /**
   * Mapped allocations. This uses atomic_allocator to ensure that the keys of
   * the map do not prevent garbage collection.
   */
  std::unordered_map<pointer_type,pointer_type,hash_type,equal_to_type,
      atomic_allocator<std::pair<const pointer_type,pointer_type>>> map;

  /**
   * Values of the map are replicated here to ensure that they do prevent
   * garbage collection.
   */
  std::list<pointer_type,gc_allocator<pointer_type>> values;

  /**
   * Generation.
   */
  size_t gen;
};
}
