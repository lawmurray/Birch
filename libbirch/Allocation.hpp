/**
 * @file
 */
#pragma once

#include "libbirch/SharedPointer.hpp"
#include "libbirch/WeakPointer.hpp"

namespace bi {
class Any;

/**
 * Control structure for allocations.
 *
 * @ingroup libbirch
 */
class Allocation {
public:
  /**
   * Constructor.
   *
   * @param parent Parent for lazy copy.
   * @param world World number.
   */
  Allocation(Allocation* parent, const world_t world);

  /**
   * Constructor.
   *
   * @param object Managed object.
   */
  Allocation(Any* object = nullptr);

  /**
   * Destructor.
   */
  ~Allocation();

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
  const world_t world;

private:
  /**
   * Detach the parent.
   */
  void detach();

  /**
   * Deallocate the managed object, if any. This occurs when the share count
   * is zero.
   */
  void deallocate();

  /**
   * Destroy the allocation object. This occurs when both the share and weak
   * counts are zero.
   */
  void destroy();

  /**
   * Parent for lazy copy.
   */
  Allocation* parent;

  /**
   * Managed object.
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
