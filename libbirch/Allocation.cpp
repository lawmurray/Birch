/**
 * @file
 */
#include "libbirch/Allocation.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/global.hpp"

#include <cassert>

bi::Allocation::Allocation(Allocation* parent) :
    world(fiberWorld),
    parent(parent),
    object(nullptr),
    shared(1),
    weak(0) {
  parent->sharedInc();
}

bi::Allocation::Allocation(Any* object) :
    world(fiberWorld),
    parent(nullptr),
    object(object),
    shared(1),
    weak(0) {
  //
}

bi::Allocation::~Allocation() {
  assert(shared == 0);
  assert(weak == 0);
}

bi::Allocation* bi::Allocation::make(Allocation* parent) {
  return new Allocation(parent);
}

bi::Allocation* bi::Allocation::make(Any* object) {
  return new Allocation(object);
}

bi::Any* bi::Allocation::get() {
  if (world != fiberWorld) {
    /* the object needs to be imported into this world; it may have already
     * been imported via another pointer into this world, or an intermediate
     * world between this world and the object world on the world tree; first
     * apply these */
    assert(parent);
    object = parent->get();
    parent->sharedDec();
    parent = nullptr;

    object = allocationMap.get(this);
    object = object->clone();
  }
  return object;
}

void bi::Allocation::sharedInc() {
  ++shared;
}

void bi::Allocation::sharedDec() {
  --shared;
  assert(shared >= 0);
  if (shared == 0) {
    delete object;
    object = nullptr;
    if (weak == 0) {
      if (parent) {
        parent->sharedDec();
      }
      delete this;
    }
  }
}

void bi::Allocation::weakInc() {
  ++weak;
}

void bi::Allocation::weakDec() {
  --weak;
  assert(weak >= 0);
  if (shared == 0 && weak == 0) {
    if (parent) {
      parent->sharedDec();
    }
    delete this;
  }
}

size_t std::hash<bi::Allocation>::operator()(const bi::Allocation& o) const {
  /* We construct a 64-bit integer as follows, and then apply std::hash to
   * it. For the raw pointer, the lower 4-bits are considered irrelevant due
   * to alignment, and higher bits may have low entropy. For the world
   * number, lower bits have higher entropy. So we shift away the higher 16
   * bits of the raw pointer and copy in the lower 20 bits of the world
   * number, overwriting the lower 4 bits of the original raw pointer in the
   * process. We then trust std::hash on the result. */
  uint64_t value = ((uint64_t)o.object << 16) | ((uint64_t)o.world & 0xFFFFF);
  return std::hash<uint64_t>::operator()(value);
}

bool std::equal_to<bi::Allocation>::operator()(const bi::Allocation& o1,
    const bi::Allocation& o2) const {
  return o1.object == o2.object && o1.world == o2.world;
}
