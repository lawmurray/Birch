/**
 * @file
 */
#include "libbirch/AllocationMap.hpp"

#include "libbirch/Allocation.hpp"
#include "libbirch/Any.hpp"

#include <cassert>

bi::Any* bi::AllocationMap::get(Allocation* from) const {
  auto to = map.find(from);
  if (to != map.end()) {
    return to->second;
  } else {
    return from->object;
  }
}

void bi::AllocationMap::set(Allocation* from, Any* to) {
  auto result = map.insert(std::make_pair(from, to));
  assert(result.second);
}
