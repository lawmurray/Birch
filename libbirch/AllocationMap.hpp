/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

#include <unordered_map>

namespace std {
template<>
struct hash<std::pair<bi::Allocation*,bi::world_t>> : public std::hash<
    uint64_t> {
  size_t operator()(const std::pair<bi::Allocation*,bi::world_t>& o) const;
};

template<>
struct equal_to<std::pair<bi::Allocation*,bi::world_t>> {
  bool operator()(const std::pair<bi::Allocation*,bi::world_t>& o1,
      const std::pair<bi::Allocation*,bi::world_t>& o2) const;
};
}

namespace bi {
class Allocation;

/**
 * Allocation map for fiber.
 *
 * @ingroup libbirch
 */
class AllocationMap {
public:
  /**
   * Retrieve (and possibly create) an allocation.
   *
   * @param src The source allocation.
   * @param world The world into which the source allocation is being
   * imported.
   *
   * @return The allocation for this import. If no such allocation exists yet,
   * one is created and returned.
   */
  Allocation* get(Allocation* src, const world_t world);

  /**
   * Insert a mapping.
   *
   * @param src The source allocation.
   * @param world The world into which the source allocation is being
   * imported.
   * @param dst The resulting allocation.
   */
  void insert(Allocation* src, const world_t world, Allocation* dst);

  /**
   * Remove all mappings for a given allocation. This is used whem the
   * allocation is deleted.
   *
   * @param src The source allocation.
   */
  void remove(Allocation* src);

private:
  /**
   * Mapped allocations.
   */
  std::unordered_map<std::pair<Allocation*,world_t>,Allocation*> map;

  /**
   * Record of the worlds into which each allocation has been imported.
   */
  std::unordered_multimap<Allocation*,world_t> imports;
};
}
