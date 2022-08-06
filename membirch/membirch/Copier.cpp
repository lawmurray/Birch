/**
 * @file
 */
#include "membirch/Copier.hpp"

#include "membirch/Any.hpp"

membirch::Any* membirch::Copier::visitObject(Any* o) {
  auto& value = m.get(o);
  if (value) {
    return value;
  } else {
    value = o->copy_();

    /* copy the value into a non-reference, as the reference may be
     * invalidated if m is resized during the call to accept_() below */
    Any* result = value;
    result->accept_(*this);
    return result;
  }
}
