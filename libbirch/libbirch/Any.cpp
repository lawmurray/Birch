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

libbirch::Any::Any(const Any& o) :
    r_(0),
    a_(0),
    l_(std::numeric_limits<int>::max()),
    h_(0),
    p_(-1),
    f_(o.f_.load() & ACYCLIC) {
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

  if ((old & HEAD) && r == a_ - 1) {
    /* last external reference about to be removed, remainder are internal
     * to the biconnected component; can collect the whole biconnected
     * component now */
    biconnected_collect(this);
  }
  if (r == 0) {
    destroy_();
    if (old & BUFFERED && !(old & ACYCLIC)) {
      /* this is currently registered as a possible root, attempt to
       * simultaneously deregister and deallocate it */
      deregister_possible_root(this);
    } else {
      /* otherwise, this is not currently registered as a possible root, and
       * can be deallocated immediately */
      assert(!contains_possible_root(this));
      deallocate_();
    }
  } else if (!(old & BUFFERED) && !(old & ACYCLIC)) {
    /* not already registered as a possible root, and not acyclic nor a bridge
     * head, so register as possible root of cycle */
    register_possible_root(this);
  }
}

void libbirch::Any::decSharedBiconnected_() {
  assert(numShared_() > 0);

  auto r = --r_;
  auto old = f_.exchangeOr(BUFFERED|POSSIBLE_ROOT);

  if (r == 0) {
    destroy_();
    if (old & BUFFERED && !(old & ACYCLIC)) {
      /* this is currently registered as a possible root, attempt to
       * simultaneously deregister and deallocate it */
      deregister_possible_root(this);
    } else {
      /* otherwise, this is not currently registered as a possible root, and
       * can be deallocated immediately */
      assert(!contains_possible_root(this));
      deallocate_();
    }
  }
}
