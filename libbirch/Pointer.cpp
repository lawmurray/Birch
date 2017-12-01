/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/global.hpp"
#include "libbirch/AllocationMap.hpp"
#include "libbirch/FiberWorld.hpp"

#include <cassert>

size_t std::hash<bi::Pointer<bi::Any>>::operator()(
    const bi::Pointer<bi::Any>& o) const {
  /* We construct a 64-bit integer as follows, and then apply std::hash to
   * it. For the raw pointer, the lower 4-bits are considered irrelevant due
   * to alignment, and higher bits may have low entropy. For the world
   * number, lower bits have higher entropy. So we shift away the higher 16
   * bits of the raw pointer and copy in the lower 20 bits of the world
   * number, overwriting the lower 4 bits of the original raw pointer in the
   * process. We then trust std::hash on the result. */
  int64_t value = ((int64_t)o.raw << 16) | ((int64_t)o.world->id & 0xFFFFF);
  return std::hash<int64_t>::operator()(value);
}

bool std::equal_to<bi::Pointer<bi::Any>>::operator()(
    const bi::Pointer<bi::Any>& o1, const bi::Pointer<bi::Any>& o2) const {
  return o1.raw == o2.raw && o1.world == o2.world;
}

bi::Pointer<bi::Any>::Pointer(Any* raw) :
    world(fiberWorld),
    raw(raw) {
  //
}

bi::Pointer<bi::Any>& bi::Pointer<bi::Any>::operator=(Any* raw) {
  assert(world == fiberWorld);
  this->raw = raw;
  return *this;
}

bool bi::Pointer<bi::Any>::isNull() const {
  return !raw;
}

bi::Any* bi::Pointer<bi::Any>::get() {
  if (world != raw->world) {
    /* the object needs to be imported into this world; it may have already
     * been imported via another pointer into this world, or an intermediate
     * world between this world and the object world on the world tree; first
     * apply these */
    FiberWorld* toWorld = world;
    Any* to;
    do {
      to = allocationMap->get(*this);
      if (to == raw) {
        world = parent->world;
      } else {
        raw = to;
        world = toWorld;
      }
    } while (world != raw->world);

    if (world != toWorld) {
      /* copy and import */
      to = raw->clone();
      world = toWorld;
      allocationMap->set(*this, to);
      raw = to;
    }
  }
  return raw;
}
