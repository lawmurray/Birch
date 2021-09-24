/**
 * @file
 */
#include "libbirch/Any.hpp"
#include "libbirch/Destroyer.hpp"

libbirch::Any::Any() :
    r_(0),
    a_(0),
    l_(std::numeric_limits<int>::max()),
    h_(0),
    p_(-1),
    f_(0) {
  //
}

libbirch::Any::Any(const Any& o) : Any() {
  //  
}

void libbirch::Any::destroy_() {
  Destroyer v;
  this->accept_(v);
}

void libbirch::Any::decShared_() {
  assert(numShared_() > 0);

  auto r = --r_;
  auto old = f_.exchangeOr(BUFFERED|POSSIBLE_ROOT);
  if (r == 0) {
    destroy_();
    if (old & BUFFERED) {
      /* this is currently registered as a possible root, attempt to
       * simultaneously deregister and deallocate it */
      deregister_possible_root(this);
    } else {
      /* otherwise, this is not currently registered as a possible root, and
       * can be deallocated immediately */
      deallocate_();
    }
  } else if (!(old & BUFFERED)) {
    /* not already registered as a possible root, register now */
    register_possible_root(this);
  }
}

void libbirch::Any::decSharedBiconnected_() {
  assert(numShared_() > 0);

  auto r = --r_;
  if (r == 0) {
    destroy_();

    auto old = f_.load();
    if (old & BUFFERED) {
      /* currently registered; could attempt to deregister here, but this is
       * only successful for recently-registered objects, which is highly
       * unlikely in this scenario, so don't bother trying */
      //deregister_possible_root(this);
    } else {
      /* not currently registered, can deallocate immediately */
      deallocate_();
    }
  }
}

void libbirch::Any::decSharedBridge_() {
  assert(numShared_() > 0);

  auto r = --r_;
  if (r == a_ - 1) {
    /* last external reference just removed, remainder are internal to the
     * biconnected component; collect the whole biconnected component now */

    /* have to be careful here; during biconnected_collect() we decrement the
     * reference count on each object *after* visiting, to avoid destroying it
     * while the visit is in progress, but for the head node we decrement
     * *before* in order to trigger the whole visit; this means an internal
     * edge that decrements the head node reference count to zero will destroy
     * it before the visit has finished; to avoid, we restore the count first,
     * then visit, then decrement again */
    r_.increment();
    biconnected_collect(this);
    r_.decrement();
    assert(r_.load() == 0);
    destroy_();

    /* head nodes are not registered as possible roots *after* they are
     * designated as such, but can be registered *before* they are designated
     * as such */
    auto old = f_.load();
    if (old & BUFFERED) {
      /* currently registered; could attempt to deregister here, but this is
       * only successful for recently-registered objects, which is highly
       * unlikely in this scenario, so don't bother trying */
      //deregister_possible_root(this);
    } else {
      /* not currently registered, can deallocate immediately */
      deallocate_();
    }
  }
}

void libbirch::Any::decSharedAcyclic_() {
  assert(numShared_() > 0);
  assert(isAcyclic_());

  auto r = --r_;
  if (r == 0) {
    destroy_();

    /* acyclic objects are so for life, and are never registered as possible
     * roots; can deallocate immediately */
    deallocate_();
  }
}
