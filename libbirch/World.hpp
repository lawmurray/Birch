/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Lockable.hpp"

#include <unordered_map>

namespace bi {
/**
 * Fiber world.
 *
 * @ingroup libbirch
 */
class World: public Counted, private Lockable {
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
  SharedPtr<Any> get(const SharedPtr<Any>& o, World* world);

  /**
   * Get an object.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  SharedPtr<Any> getNoCopy(const SharedPtr<Any>& o, World* world);

private:
  /**
   * Pull and copy (if necessary) an object from a clone ancestor into this
   * world.
   *
   * @param o The object.
   *
   * @return The mapped and copied object.
   */
  SharedPtr<Any> pull(const SharedPtr<Any>& o, World* world);

  /**
   * Pull an object from a clone ancestor into this world.
   *
   * @param src The source object.
   *
   * @return The mapped object.
   */
  SharedPtr<Any> pullNoCopy(const SharedPtr<Any>& o,
      World* world);

  /**
   * The world from which this world was cloned.
   */
  SharedPtr<World> cloneSource;

  /**
   * The world from which this world was launched.
   */
  World* launchSource;

  /**
   * Custom hash function for maps.
   */
  template<class T>
  struct world_hash {
    std::size_t operator()(const T& o) const {
      return reinterpret_cast<std::size_t>(o) >> 6;
    }
  };


  /*
   * Type of map.
   */
  using map_key_type = Any*;
  using map_value_type = SharedPtr<Any>;
  using map_hash_type = world_hash<map_key_type>;
  using map_equal_type = std::equal_to<map_key_type>;
  using map_alloc_type = Allocator<std::pair<const map_key_type,
      map_value_type>>;
  using map_type = std::unordered_map<map_key_type,map_value_type,
      map_hash_type,map_equal_type,map_alloc_type>;

  /**
   * Mapped allocations.
   */
  map_type map;

  /*
   * Type of cache.
   */
  using cache_key_type = Any*;
  using cache_value_type = Any*;
  using cache_hash_type = world_hash<cache_key_type>;
  using cache_equal_type = std::equal_to<cache_key_type>;
  using cache_alloc_type = Allocator<std::pair<const cache_key_type,
      cache_value_type>>;
  using cache_type = std::unordered_map<cache_key_type,cache_value_type,
      cache_hash_type,cache_equal_type,cache_alloc_type>;

  /**
   * Cached mappings of clone ancestors.
   */
  cache_type cache;

  /**
   * Launch depth.
   */
  int launchDepth;
};
}
