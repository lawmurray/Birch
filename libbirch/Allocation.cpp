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
  assert(parent);
  parent->sharedInc();
}

bi::Allocation::Allocation(Any* object) :
    world(fiberWorld),
    parent(nullptr),
    object(object),
    shared(1),
    weak(0) {
  assert(object);
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
    assert(world == fiberWorld);  // shouldn't change in clone
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
  assert(shared > 0);
  --shared;
  if (shared == 0) {
    /* as each object holds a weak pointer to itself, the weak count should
     * never be zero at this point; if the last weak pointer is that held by
     * the object itself, the destructor of the object will in turn call
     * weakDec() to clean up the rest */
    assert(weak > 0);
    detach();
    deallocate();
  }
}

uint32_t bi::Allocation::weakCount() const {
  return weak;
}

void bi::Allocation::weakInc() {
  ++weak;
}

void bi::Allocation::weakDec() {
  assert(weak > 0);
  --weak;
  if (weak == 0 && shared == 0) {
    /* as each object holds a weak pointer to itself, the weak count should
     * never be zero unless the object has been destroyed, in which case the
     * shared count should be zero too */
    //assert(shared == 0);  // not sure, what if object isn't created yet?
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
    /* if the last weak pointer is held by the object itself, deleting the
     * object will call destroy(), which will delete this object; in case this
     * happens, we should not write nullptr to this object, so copy to
     * temporary first */
    auto tmp = object;
    object = nullptr;
    delete tmp;
  }
}

void bi::Allocation::destroy() {
  allocationMap.remove(this);
  delete this;
}
