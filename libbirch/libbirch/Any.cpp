/**
 * @file
 */
#include "libbirch/Any.hpp"

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
    if (!(old & BUFFERED) || old & ACYCLIC || old & HEAD) {
      /* hasn't been previously buffered, is acyclic, or is the head of a
        * biconnected component, and so can be immediately deallocated */
      deallocate_();
    }
  } else if (!(old & BUFFERED) && !(old & ACYCLIC) && !(old & HEAD)) {
    /* not already registered as a possible root, and not acyclic */
    register_possible_root(this);
  }
}
