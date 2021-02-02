/**
 * @file
 */
#include "libbirch/Collector.hpp"

void libbirch::Collector::visit(Any* o) {
  auto old = o->Any::f.exchangeOr(COLLECTED);
  if (!(old & COLLECTED) && !(old & REACHED)) {
    register_unreachable(o);
    o->accept_(*this);
  }
}
