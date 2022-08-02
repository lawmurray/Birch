/**
 * @file
 */
#include "libbirch/Marker.hpp"

#include "libbirch/Any.hpp"

void libbirch::Marker::visitObject(Any* o) {
  if (!(o->f_.exchangeOr(MARKED) & MARKED)) {
    o->f_.maskAnd(~(POSSIBLE_ROOT|BUFFERED|SCANNED|REACHED|COLLECTED));
    o->accept_(*this);
  }
}
