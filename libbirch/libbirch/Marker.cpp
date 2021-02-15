/**
 * @file
 */
#include "libbirch/Marker.hpp"

void libbirch::Marker::visit(Any* o) {
  if (!(o->f_.exchangeOr(MARKED) & MARKED)) {
    o->f_.maskAnd(~(POSSIBLE_ROOT|BUFFERED|SCANNED|REACHED|COLLECTED));
    o->accept_(*this);
  }
}
