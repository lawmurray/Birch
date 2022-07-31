/**
 * @file
 */
#include "libbirch/BiconnectedCopier.hpp"

#include "libbirch/Any.hpp"

libbirch::Any* libbirch::BiconnectedCopier::visitObject(Any* o) {
  auto& value = m.get(o);
  if (!value) {
    value = o->copy_();
    value->accept_(*this);
  }
  return value;
}
