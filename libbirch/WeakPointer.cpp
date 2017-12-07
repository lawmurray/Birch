/**
 * @file
 */
#include "libbirch/WeakPointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/SharedPointer.hpp"

#include <cassert>

bi::WeakPointer<bi::Any>::WeakPointer() :
    allocation(nullptr) {
  //
}

bi::WeakPointer<bi::Any>::WeakPointer(Allocation* allocation) :
    allocation(allocation) {
  assert(allocation);
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>::WeakPointer(const WeakPointer<Any>& o) {
  if (o.allocation) {
    allocation = allocationMap.get(o.allocation);
    allocation->weakInc();
  } else {
    allocation = nullptr;
  }
}

bi::WeakPointer<bi::Any>::WeakPointer(const SharedPointer<Any>& o) {
  if (o.allocation) {
    allocation = allocationMap.get(o.allocation);
    allocation->weakInc();
  } else {
    allocation = nullptr;
  }
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  assert(!allocation || allocation->world == fiberWorld);
  reset(allocationMap.get(o.allocation));
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  assert(!allocation || allocation->world == fiberWorld);
  reset(allocationMap.get(o.allocation));
  return *this;
}

bi::WeakPointer<bi::Any>::~WeakPointer() {
  release();
}

bool bi::WeakPointer<bi::Any>::query() const {
  assert(!allocation || allocation->weakCount() > 0);
  return allocation && allocation->sharedCount() > 0;
}

bi::Any * bi::WeakPointer<bi::Any>::get() const {
  assert(allocation);
  return allocation->get();
}

void bi::WeakPointer<bi::Any>::reset(Allocation* allocation) {
  if (this->allocation != allocation) {
    if (this->allocation) {
      this->allocation->weakDec();
    }
    this->allocation = allocation;
    if (this->allocation) {
      this->allocation->weakInc();
    }
  }
}

void bi::WeakPointer<bi::Any>::release() {
  if (allocation) {
    allocation->weakDec();
    allocation = nullptr;
  }
}
