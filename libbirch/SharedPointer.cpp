/**
 * @file
 */
#include "libbirch/SharedPointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/WeakPointer.hpp"

#include <cassert>

bi::SharedPointer<bi::Any>::SharedPointer(Any* raw) {
  allocation = Allocation::make(raw);
}

bi::SharedPointer<bi::Any>::SharedPointer(const SharedPointer<Any>& o) {
  /* pointers are only copied when objects are cloned; objects are only cloned
   * when being moved between worlds (e.g. copy-on-write for fibers) */
  if (o.allocation) {
    allocation = Allocation::make(o.allocation);
  } else {
    allocation = nullptr;
  }
}

bi::SharedPointer<bi::Any>::SharedPointer(const WeakPointer<Any>& o) :
    allocation(o.allocation) {
  allocation->sharedInc();
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  if (allocation) {
    release();
  }
  if (o.allocation) {
    assert(o.allocation->world == fiberWorld);
    allocation = o.allocation;
    allocation->sharedInc();
  }
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  if (allocation) {
    release();
  }
  if (o.allocation) {
    assert(o.allocation->world == fiberWorld);
    allocation = o.allocation;
    allocation->sharedInc();
  }
  return *this;
}

bi::SharedPointer<bi::Any>::~SharedPointer() {
  release();
}

bool bi::SharedPointer<bi::Any>::query() const {
  return allocation;
}

bi::Any* bi::SharedPointer<bi::Any>::get() const {
  return allocation ? allocation->get() : nullptr;
}

void bi::SharedPointer<bi::Any>::release() {
  if (allocation) {
    allocation->sharedDec();
    allocation = nullptr;
  }
}

bi::SharedPointer<bi::Any>::SharedPointer(Allocation* allocation) :
    allocation(allocation) {
  //
}
