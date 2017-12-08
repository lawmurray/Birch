/**
 * @file
 */
#include "libbirch/SharedPointer.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"
#include "libbirch/WeakPointer.hpp"

#include <cassert>

bi::SharedPointer<bi::Any>::SharedPointer(const std::nullptr_t) :
    allocation(nullptr) {
  //
}

bi::SharedPointer<bi::Any>::SharedPointer(Any* raw) {
  if (raw) {
    allocation = new Allocation(raw);
    assert(allocation->sharedCount() == 1);
  } else {
    allocation = nullptr;
  }
}

bi::SharedPointer<bi::Any>::SharedPointer(Allocation* allocation) :
    allocation(allocation) {
  if (allocation) {
    allocation->sharedInc();
  }
}

bi::SharedPointer<bi::Any>::SharedPointer(const SharedPointer<Any>& o) {
  if (o.allocation) {
    allocation = allocationMap.get(o.allocation);
    allocation->sharedInc();
  } else {
    allocation = nullptr;
  }
}

bi::SharedPointer<bi::Any>::SharedPointer(SharedPointer<Any>&& o) {
  if (o.allocation) {
    allocation = allocationMap.get(o.allocation);
    allocation->sharedInc();
  } else {
    allocation = nullptr;
  }
}

bi::SharedPointer<bi::Any>::SharedPointer(const WeakPointer<Any>& o) {
  if (o.allocation) {
    allocation = allocationMap.get(o.allocation);
    allocation->sharedInc();
  } else {
    allocation = nullptr;
  }
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  release();
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  reset(allocationMap.get(o.allocation));
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    SharedPointer<Any>&& o) {
  reset(allocationMap.get(o.allocation));
  return *this;
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  reset(allocationMap.get(o.allocation));
  return *this;
}

bi::SharedPointer<bi::Any>::~SharedPointer() {
  release();
}

bool bi::SharedPointer<bi::Any>::query() const {
  assert(!allocation || allocation->sharedCount() > 0);
  return allocation;
}

bi::Any* bi::SharedPointer<bi::Any>::get() const {
  assert(allocation);
  return allocation->get();
}

void bi::SharedPointer<bi::Any>::reset(Any* raw) {
  release();
  if (raw) {
    allocation = new Allocation(raw);
    assert(allocation->sharedCount() == 1);
  }
}

void bi::SharedPointer<bi::Any>::reset(Allocation* allocation) {
  if (this->allocation != allocation) {
    if (this->allocation) {
      this->allocation->sharedDec();
    }
    this->allocation = allocation;
    if (this->allocation) {
      this->allocation->sharedInc();
    }
  }
}

void bi::SharedPointer<bi::Any>::release() {
  if (allocation) {
    allocation->sharedDec();
    allocation = nullptr;
  }
}
