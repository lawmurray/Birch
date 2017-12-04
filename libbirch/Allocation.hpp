/**
 * @file
 */
#pragma once

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
   * @param object The object.
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
   * @param object The object.
   */
  static Allocation* make(Any* object);

  /**
   * Get the object.
   */
  Any* get();

  /**
   * Increment shared count.
   */
  void sharedInc();

  /**
   * Decrement shared count.
   */
  void sharedDec();

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
