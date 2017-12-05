/**
 * @file
 */
#pragma once

#include "libbirch/SharedPointer.hpp"
#include "libbirch/WeakPointer.hpp"

#include <unordered_map>

namespace bi {
class Any;

/**
 * Control structure for allocations.
 *
 * @ingroup libbirch
 */
class Allocation {
  friend struct std::hash<Allocation>;
  friend struct std::equal_to<Allocation>;
  friend class AllocationMap;
private:
  /**
   * Constructor.
   *
   * @param parent Parent allocation for copy-on-write.
   */
  Allocation(Allocation* parent);

  /**
   * Constructor.
   *
   * @param object The object to manage.
   */
  Allocation(Any* object);

  /**
   * Destructor.
   */
  ~Allocation();

public:
  /**
   * Factory method.
   *
   * @param parent Parent allocation for copy-on-write.
   */
  static Allocation* make(Allocation* parent);

  /**
   * Factory method.
   *
   * @param object The object to manage.
   */
  static Allocation* make(Any* object);

  /**
   * Get the object.
   */
  Any* get();

  /**
   * The current shared count.
   */
  uint32_t sharedCount() const;

  /**
   * Increment shared count.
   */
  void sharedInc();

  /**
   * Decrement shared count.
   */
  void sharedDec();

  /**
   * The current weak count.
   */
  uint32_t weakCount() const;

  /**
   * Increment weak count.
   */
  void weakInc();

  /**
   * Decrement weak count.
   */
  void weakDec();

  /**
   * World number.
   */
  const uint64_t world;

private:
  /**
   * Deallocate the managed object, if any. This occurs when the share count
   * is zero.
   */
  void deallocate();

  /**
   * Detach the deallocation object from its parent, if any. This occurs when
   * the managed object is copied so that the parent is no longer needed, or
   * when the object is destroyed.
   */
  void detach();

  /**
   * Destroy the allocation object. This occurs when both the share and weak
   * counts are zero.
   */
  void destroy();

  /**
   * Parent allocation for copy-on-write.
   */
  Allocation* parent;

  /**
   * The allocation.
   */
  Any* object;

  /**
   * Shared count.
   */
  uint32_t shared;

  /**
   * Weak count.
   */
  uint32_t weak;
};
}

namespace std {
template<>
struct hash<bi::Allocation> : public std::hash<uint64_t> {
  size_t operator()(const bi::Allocation& o) const;
};

template<>
struct equal_to<bi::Allocation> {
  bool operator()(const bi::Allocation& o1, const bi::Allocation& o2) const;
};
}
