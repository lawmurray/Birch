/**
 * @file
 */
#include "bi/lib/AllocationMap.hpp"

#include "bi/lib/Any.hpp"

bi::AllocationMap::AllocationMap() :
    gen(0) {
  //
}

bi::AllocationMap::AllocationMap(const AllocationMap& o) :
    map(o.map),
    gen(++const_cast<AllocationMap&>(o).gen) {
  //
}

bi::AllocationMap& bi::AllocationMap::operator=(const AllocationMap& o) {
  map = o.map;
  gen = ++const_cast<AllocationMap&>(o).gen;
  return *this;
}

bi::Any* bi::AllocationMap::get(Any* from) {
  auto result = from;
  auto iter = map.find(result);
  while (iter != map.end()) {
    result = iter->second;
    iter = map.find(result);
  }
  return result;
}

void bi::AllocationMap::set(Any* from, Any* to) {
  map.insert(std::make_pair(from, to));
}
