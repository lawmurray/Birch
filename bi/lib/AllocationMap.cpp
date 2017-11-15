/**
 * @file
 */
#include "bi/lib/AllocationMap.hpp"

#include "bi/lib/Any.hpp"

#include <algorithm>

bi::AllocationMap::AllocationMap() :
    gen(0) {
  //
}

bi::AllocationMap::AllocationMap(const AllocationMap& o) :
    keys(o.keys),
    values(o.values),
    gen(++const_cast<AllocationMap&>(o).gen) {
  //
}

bi::AllocationMap& bi::AllocationMap::operator=(const AllocationMap& o) {
  keys = o.keys;
  values = o.values;
  gen = ++const_cast<AllocationMap&>(o).gen;
  return *this;
}

bi::AllocationMap* bi::AllocationMap::clone() {
  return new (GC) AllocationMap(*this);
}

bi::Any* bi::AllocationMap::get(Any* from) {
  auto result = from;
  auto iter = std::find(keys.begin(), keys.end(), result);
  while (iter != keys.end()) {
    result = values[std::distance(keys.begin(), iter)];
    iter = std::find(keys.begin(), keys.end(), result);
  }
  return result;
}

void bi::AllocationMap::set(Any* from, Any* to) {
  auto iter = std::lower_bound(keys.begin(), keys.end(), from);
  values.insert(values.begin() + std::distance(keys.begin(), iter), to);
  keys.insert(iter, from);  // invalidates iter, so do after values.insert()
}
