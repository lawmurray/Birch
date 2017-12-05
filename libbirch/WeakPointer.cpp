/**
 * @file
 */
#include "libbirch/WeakPointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/SharedPointer.hpp"

#include <cassert>

bi::WeakPointer<bi::Any>::WeakPointer(Any* raw) {
  allocation = Allocation::make(raw);
}

bi::WeakPointer<bi::Any>::WeakPointer(const SharedPointer<Any>& o) :
    allocation(o.allocation) {
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>::WeakPointer(const WeakPointer<Any>& o) {
  /* pointers are only copied when objects are cloned; objects are only cloned
   * when being moved between worlds (e.g. copy-on-write for fibers) */
  if (o.allocation) {
    allocation = Allocation::make(o.allocation);
  } else {
    allocation = nullptr;
  }
}

bi::WeakPointer<bi::Any>::~WeakPointer() {
  release();
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  if (allocation) {
    release();
  }
  if (o.allocation) {
    assert(o.allocation->world == fiberWorld);
    allocation = o.allocation;
    allocation->weakInc();
  }
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  if (allocation) {
    release();
  }
  if (o.allocation) {
    assert(o.allocation->world == fiberWorld);
    allocation = o.allocation;
    allocation->weakInc();
  }
  return *this;
}

bool bi::WeakPointer<bi::Any>::query() const {
  return allocation && allocation->sharedCount() > 0;
}

bi::Any* bi::WeakPointer<bi::Any>::get() const {
  return allocation ? allocation->get() : nullptr;
}

void bi::WeakPointer<bi::Any>::release() {
  if (allocation) {
    allocation->weakDec();
    allocation = nullptr;
  }
}

bi::WeakPointer<bi::Any>::WeakPointer(Allocation* allocation) :
    allocation(allocation) {
  //
}
