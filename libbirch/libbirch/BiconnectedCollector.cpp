/**
 * @file
 */
#include "libbirch/BiconnectedCollector.hpp"

#include "libbirch/Any.hpp"

void libbirch::BiconnectedCollector::visitObject(Any* o) {
  auto old = o->f_.exchangeOr(COLLECTED);
  if (!(old & COLLECTED)) {
    o->accept_(*this);
  }
}
