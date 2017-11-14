/**
 * @file
 */
#include "bi/lib/Any.hpp"

#include "bi/lib/global.hpp"
#include "bi/lib/AllocationMap.hpp"

bi::Any::Any() :
    gen(fiberAllocationMap ? fiberAllocationMap->gen : 0) {
  //
}

bi::Any::Any(const Any& o) :
    gen(fiberAllocationMap ? fiberAllocationMap->gen : 0) {
  //
}

bi::Any::~Any() {
  //
}

bool bi::Any::isShared() const {
  return fiberAllocationMap && gen < fiberAllocationMap->gen;
}
