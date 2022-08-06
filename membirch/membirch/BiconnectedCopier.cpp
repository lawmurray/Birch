/**
 * @file
 */
#include "membirch/BiconnectedCopier.hpp"

#include "membirch/Any.hpp"

membirch::Any* membirch::BiconnectedCopier::visitObject(Any* o) {
  auto& value = m.get(o);
  if (!value) {
    value = o->copy_();
    value->accept_(*this);
  }
  return value;
}
