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
    weak(1) {
  parent->sharedInc();
}

bi::Allocation::Allocation(Any* object) :
    world(fiberWorld),
    parent(nullptr),
    object(object),
    shared(1),
    weak(1) {
  object->ptr.reset(this);
}

bi::Allocation::~Allocation() {
  assert(shared == 0);
  assert(weak == 0);
}

bi::Any* bi::Allocation::get() {
  if (parent) {
    world_t prevWorld = fiberWorld;
    fiberWorld = world;
    object = parent->get()->clone();
    fiberWorld = prevWorld;
    detach();
    object->ptr.reset(this);
  }
  return object;
}

uint32_t bi::Allocation::sharedCount() const {
  return shared;
}

void bi::Allocation::sharedInc() {
  assert(shared > 0);
  // ^ can't restore a released resoure, e.g. via a weak pointer to the same
  ++shared;
}

void bi::Allocation::sharedDec() {
  --shared;
  assert(shared >= 0);
  if (shared == 0) {
    detach();
    deallocate();
    if (weak == 0) {
      destroy();
    }
  }
}

uint32_t bi::Allocation::weakCount() const {
  return weak;
}

void bi::Allocation::weakInc() {
  ++weak;
}

void bi::Allocation::weakDec() {
  --weak;
  assert(weak >= 0);
  if (shared == 0 && weak == 0) {
    destroy();
  }
}

void bi::Allocation::detach() {
  if (parent) {
    parent->sharedDec();
    parent = nullptr;
  }
}

void bi::Allocation::deallocate() {
  if (object) {
    delete object;
    object = nullptr;
  }
}

void bi::Allocation::destroy() {
  allocationMap.remove(this);
  delete this;
}
