/**
 * @file
 */
#include "membirch/Marker.hpp"

#include "membirch/Any.hpp"

void membirch::Marker::visitObject(Any* o) {
  if (!(o->f_.exchangeOr(MARKED) & MARKED)) {
    o->f_.maskAnd(~(POSSIBLE_ROOT|BUFFERED|SCANNED|REACHED|COLLECTED));
    o->accept_(*this);
  }
}
