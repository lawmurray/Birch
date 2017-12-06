/**
 * @file
 */
#include "libbirch/AllocationMap.hpp"

#include "libbirch/Allocation.hpp"
#include "libbirch/Any.hpp"

#include <cassert>

size_t std::hash<std::pair<bi::Allocation*,bi::world_t>>::operator()(
    const std::pair<bi::Allocation*,bi::world_t>& o) const {
  /* We construct a 64-bit integer as follows, and then apply std::hash to
   * it. For the raw pointer, the lower 4-bits are considered irrelevant due
   * to alignment, and higher bits may have low entropy. For the world
   * number, lower bits have higher entropy. So we shift away the higher 16
   * bits of the raw pointer and copy in the lower 20 bits of the world
   * number, overwriting the lower 4 bits of the original raw pointer in the
   * process. We then trust std::hash on the result. */
  uint64_t value = (reinterpret_cast<uint64_t>(o.first) << 16)
      | (reinterpret_cast<uint64_t>(o.second) & 0xFFFFF);
  return std::hash < uint64_t > ::operator()(value);
}

bool std::equal_to<std::pair<bi::Allocation*,bi::world_t>>::operator()(
    const std::pair<bi::Allocation*,bi::world_t>& o1,
    const std::pair<bi::Allocation*,bi::world_t>& o2) const {
  return o1.first == o2.first && o1.second == o2.second;
}

bi::Allocation* bi::AllocationMap::get(Allocation* src, const world_t world) {
  if (src->world == world) {
    return src;
  } else {
    auto pair = std::make_pair(src, world);
    auto iter = map.find(pair);
    if (iter != map.end()) {
      return iter->second;
    } else {
      auto dst = new Allocation(src, world);
      map.insert(std::make_pair(pair, dst));
      imports.insert(pair);
      return dst;
    }
  }
}

void bi::AllocationMap::insert(Allocation* src, const world_t world,
    Allocation* dst) {
  auto pair = std::make_pair(src, world);
  auto result = map.insert(std::make_pair(pair, dst));
  assert(result.second);
  imports.insert(pair);
}

void bi::AllocationMap::remove(Allocation* src) {
  auto range = imports.equal_range(src);
  for (auto iter = range.first; iter != range.second; ++iter) {
    map.erase(*iter);
  }
  imports.erase(range.first, range.second);
}
