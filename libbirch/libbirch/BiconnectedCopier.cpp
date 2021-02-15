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
    value = o->copy_();
    value->accept_(*this);
  }
  return value;
}
