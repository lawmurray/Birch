/**
 * @file
 */
#include "bi/lib/AllocationMap.hpp"

#include "bi/lib/global.hpp"
#include "bi/lib/Pointer.hpp"
#include "bi/lib/Any.hpp"

bi::AllocationMap::AllocationMap() :
    gen(0) {
  //
}

bi::AllocationMap::AllocationMap(const AllocationMap& o) :
    map(o.map),
    values(o.values),
    gen(++const_cast<AllocationMap&>(o).gen) {
}

bi::AllocationMap& bi::AllocationMap::operator=(const AllocationMap& o) {
  map = o.map;
  values = o.values;
  gen = ++const_cast<AllocationMap&>(o).gen;
  return *this;
}

bi::AllocationMap* bi::AllocationMap::clone() {
  return new (GC) AllocationMap(*this);
}

bi::Pointer<bi::Any> bi::AllocationMap::get(const Pointer<Any>& from) {
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
  values.push_back(to);
}
