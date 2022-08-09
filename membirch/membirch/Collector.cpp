/**
 * @file
 */
#include "membirch/Collector.hpp"

#include "membirch/Any.hpp"

void membirch::Collector::visitObject(Any* o) {
  if (!(o->f_.load() & REACHED)) {
    auto old = o->f_.exchangeOr(COLLECTED);
    if (!(old & COLLECTED)) {
      assert(o->numShared_() == 0);
      o->accept_(*this);
      register_unreachable(o);
    }
  }
}
