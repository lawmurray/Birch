/**
 * @file
 */
#include "libbirch/BiconnectedCopier.hpp"

#include "libbirch/Any.hpp"

libbirch::BiconnectedCopier::BiconnectedCopier(Any* o) : m(o) {
  //
}

libbirch::Any* libbirch::BiconnectedCopier::visit(Any* o) {
  auto& value = m.get(o);
  if (!value) {
    assert(!biconnected_copy());
    biconnected_copy(true);
    assert(biconnected_copy());
    value = o->copy();
    biconnected_copy(true);
    assert(!biconnected_copy());
    value->accept_(*this);
  }
  return value;
}
