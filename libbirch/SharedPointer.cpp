/**
 * @file
 */
#include "libbirch/SharedPointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/WeakPointer.hpp"

#include <cassert>

bi::SharedPointer<bi::Any>::SharedPointer(Any* raw) :
    allocation(new Allocation(raw)) {
  assert(allocation->sharedCount() == 1);
}

bi::SharedPointer<bi::Any>::SharedPointer(Allocation* allocation) :
    allocation(allocation) {
  allocation->sharedInc();
}

bi::SharedPointer<bi::Any>::SharedPointer(const SharedPointer<Any>& o) :
    allocation(o.allocation) {
  allocation->sharedInc();
}

bi::SharedPointer<bi::Any>::SharedPointer(const WeakPointer<Any>& o) :
    allocation(o.allocation) {
  allocation->sharedInc();
}

bi::SharedPointer<bi::Any>::SharedPointer(const SharedPointer<Any>& o,
    const world_t world) :
    allocation(allocationMap.get(o.allocation, world)) {
  allocation->sharedInc();
}

bi::SharedPointer<bi::Any>::SharedPointer(const WeakPointer<Any>& o,
    const world_t world) :
    allocation(allocationMap.get(o.allocation, world)) {
  allocation->sharedInc();
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  reset(allocationMap.get(o.allocation, allocation->world));
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  reset(allocationMap.get(o.allocation, allocation->world));
  return *this;
}

bi::SharedPointer<bi::Any>::~SharedPointer() {
  release();
}

bool bi::SharedPointer<bi::Any>::query() const {
  return allocation;
}

bi::Any* bi::SharedPointer<bi::Any>::get() const {
  return allocation->get();
}

void bi::SharedPointer<bi::Any>::reset(Allocation* allocation) {
  if (this->allocation != allocation) {
    this->allocation->sharedDec();
    this->allocation = allocation;
    this->allocation->sharedInc();
  }
}

void bi::SharedPointer<bi::Any>::release() {
  allocation->sharedDec();
  allocation = nullptr;
}
