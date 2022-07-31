/**
 * @file
 */
#include "libbirch/Collector.hpp"

#include "libbirch/Any.hpp"

void libbirch::Collector::visitObject(Any* o) {
  if (!(o->f_.load() & REACHED)) {
    auto old = o->f_.exchangeOr(COLLECTED);
    if (!(old & COLLECTED)) {
      assert(o->numShared_() == 0);
      o->accept_(*this);
      register_unreachable(o);
    }
  }
}
