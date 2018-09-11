/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Map.hpp"

namespace bi {
/**
 * Fiber world.
 *
 * @ingroup libbirch
 */
class World: public Counted {
public:
  /**
   * Default constructor.
   */
  World();

  /**
   * Constructor for root.
   */
  World(int);

  /**
   * Constructor for clone.
   *
   * @param cloneSource Clone parent.
   */
  World(const SharedPtr<World>& cloneSource);

  /**
   * Destructor.
   */
  virtual ~World();

  /**
   * Deallocate.
   */
  virtual void destroy();

  /**
   * Does this world have the given world as a clone ancestor?
   */
  bool hasCloneAncestor(World* world) const;

  /**
   * Does this world have the given world as a launch ancestor?
   */
  bool hasLaunchAncestor(World* world) const;

  /**
   * Get launch depth.
   */
  int depth() const;

  /**
   * Get an object, copying it if necessary.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  Any* get(Any* o);

  /**
   * Get an object.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  Any* getNoCopy(Any* o);

  /**
   * The world from which this world was cloned.
   */
  SharedPtr<World> cloneSource;

  /**
   * The world from which this world was launched.
   */
  World* launchSource;

  /**
   * Mapped allocations.
   */
  Map map;

private:
  /**
   * Launch depth.
   */
  int launchDepth;
};

/**
 * Pull and copy (if necessary) an object.
 *
 * @param o The object.
 * @param world The world to which to map.
 *
 * @return The mapped object.
 */
Any* pull(Any* o, World* world);

/**
 * Pull an object.
 *
 * @param o The object.
 * @param world The world to which to map.
 *
 * @return The mapped object.
 */
Any* pullNoCopy(Any* o, World* world);

}
