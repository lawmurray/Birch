/**
 * @file
 */
#include "bi/lib/Any.hpp"

#include "bi/lib/AllocationMap.hpp"
#include "bi/lib/global.hpp"

#include <cassert>

bi::Pointer<bi::Any>::Pointer(Any* raw) :
    raw(raw),
    gen(fiberGen) {
  //
}

bi::Pointer<bi::Any>::Pointer(Any* raw, const size_t gen) :
    raw(raw),
    gen(gen) {
  //
}

bi::Pointer<bi::Any>& bi::Pointer<bi::Any>::operator=(Any* raw) {
  this->raw = raw;
  this->gen = fiberGen;
  return *this;
}

bool bi::Pointer<bi::Any>::isNull() const {
  return !raw;
}

bi::Any* bi::Pointer<bi::Any>::get() const {
  assert(fiberAllocationMap);

  if (gen < fiberGen && raw) {
    /* object is shared; it may have been cloned already via another pointer,
     * so update this pointer via the current fiber's allocation map */
    auto from = *this;
    auto to = fiberAllocationMap->get(from);

    /* object is writeable; if it is still shared, then clone it and add a
     * new entry to the current fiber's allocation map */
    if (to.gen < fiberGen) {
      from = to;
      to = from.raw->clone();
      fiberAllocationMap->set(from, to);
    }
    *const_cast<Pointer<Any>*>(this) = to;
  }
  return raw;
}
