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
  Any* get(Any* o, World* world);

  /**
   * Get an object.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  Any* getNoCopy(Any* o, World* world);

private:
  /**
   * Pull and copy (if necessary) an object from a clone ancestor into this
   * world.
   *
   * @param o The object.
   *
   * @return The mapped and copied object.
   */
  Any* pull(Any* o, World* world);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  Any* pullNoCopy(Any* o, World* world);

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

  /**
   * Launch depth.
   */
  int launchDepth;
};
}
