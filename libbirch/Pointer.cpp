/**
 * @file
 */
#include "libbirch/Pointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/global.hpp"

#include <cassert>

bi::Pointer<bi::Any>::Pointer(Any* raw) {
  allocation = Allocation::make(raw);
}

bi::Pointer<bi::Any>::Pointer(const Pointer<Any>& o) {
  /* pointers are only copied when objects are cloned; objects are only cloned
   * when being moved between worlds (e.g. copy-on-write for fibers) */
  if (o.allocation) {
    allocation = Allocation::make(o.allocation);
  } else {
    allocation = nullptr;
  }
}

bi::Pointer<bi::Any>::~Pointer() {
  release();
}

bi::Pointer<bi::Any>& bi::Pointer<bi::Any>::operator=(const Pointer<Any>& o) {
  if (o.allocation) {
    assert(o.allocation->world == fiberWorld);
    allocation = o.allocation;
    allocation->sharedInc();
  } else {
    release();
  }
  return *this;
}

bi::Pointer<bi::Any>& bi::Pointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bool bi::Pointer<bi::Any>::isNull() const {
  return !allocation;
}

bi::Any* bi::Pointer<bi::Any>::get() const {
  return allocation ? allocation->get() : nullptr;
}

void bi::Pointer<bi::Any>::release() {
  if (allocation) {
    allocation->sharedDec();
    allocation = nullptr;
  }
}
