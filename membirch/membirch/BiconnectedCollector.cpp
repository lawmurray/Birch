/**
 * @file
 */
#include "membirch/BiconnectedCollector.hpp"

#include "membirch/Any.hpp"

void membirch::BiconnectedCollector::visitObject(Any* o) {
  auto old = o->f_.exchangeOr(COLLECTED);
  if (!(old & COLLECTED)) {
    o->accept_(*this);
  }
}
