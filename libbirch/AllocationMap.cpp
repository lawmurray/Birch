/**
 * @file
 */
#include "libbirch/AllocationMap.hpp"

#include "libbirch/global.hpp"
#include "libbirch/Pointer.hpp"
#include "libbirch/Any.hpp"

#include <cassert>

bi::AllocationMap::AllocationMap() {
  //
}

bi::AllocationMap::AllocationMap(const AllocationMap& o) :
    map(o.map)  {
  //
}

bi::AllocationMap& bi::AllocationMap::operator=(const AllocationMap& o) {
  map = o.map;
  return *this;
}

bi::AllocationMap* bi::AllocationMap::clone() {
  return new (GC) AllocationMap(*this);
}

bi::Pointer<bi::Any> bi::AllocationMap::get(const Pointer<Any>& from) const {
  auto to = from;
  auto iter = map.find(to);
  while (iter != map.end()) {
    to = iter->second;
    iter = map.find(to);
  }
  return to;
}

void bi::AllocationMap::set(const Pointer<Any>& from,
    const Pointer<Any>& to) {
  auto result = map.insert(std::make_pair(from, to));
  assert(result.second);
}
