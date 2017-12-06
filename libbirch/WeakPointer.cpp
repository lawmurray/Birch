/**
 * @file
 */
#include "libbirch/WeakPointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/SharedPointer.hpp"

#include <cassert>

bi::WeakPointer<bi::Any>::WeakPointer(Any* raw) :
    allocation(new Allocation(raw)) {
  assert(allocation->weakCount() == 1);
}

bi::WeakPointer<bi::Any>::WeakPointer(Allocation* allocation) :
    allocation(allocation) {
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>::WeakPointer(const WeakPointer<Any>& o) :
    allocation(o.allocation) {
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>::WeakPointer(const SharedPointer<Any>& o) :
    allocation(o.allocation) {
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>::WeakPointer(const WeakPointer<Any>& o,
    const world_t world) :
    allocation(allocationMap.get(o.allocation, world)) {
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>::WeakPointer(const SharedPointer<Any>& o,
    const world_t world) :
    allocation(allocationMap.get(o.allocation, world)) {
  allocation->weakInc();
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  reset(allocationMap.get(o.allocation, allocation->world));
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  reset(allocationMap.get(o.allocation, allocation->world));
  return *this;
}

bi::WeakPointer<bi::Any>::~WeakPointer() {
  release();
}

bool bi::WeakPointer<bi::Any>::query() const {
  return allocation && allocation->sharedCount() > 0;
}

bi::Any * bi::WeakPointer<bi::Any>::get() const {
  return allocation->get();
}

void bi::WeakPointer<bi::Any>::reset(Allocation* allocation) {
  if (this->allocation != allocation) {
    this->allocation->weakDec();
    this->allocation = allocation;
    this->allocation->weakInc();
  }
}

void bi::WeakPointer<bi::Any>::release() {
  allocation->weakDec();
  allocation = nullptr;
}
