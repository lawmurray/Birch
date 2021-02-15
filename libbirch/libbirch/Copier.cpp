/**
 * @file
 */
#include "libbirch/Copier.hpp"

#include "libbirch/Any.hpp"

libbirch::Any* libbirch::Copier::visit(Any* o) {
  auto& value = m.get(o);
  if (!value) {
    value = o->copy_();

    /* copy the value into a non-reference, as the reference may be
     * invalidated if m is resized during the call to accept_() below */
    Any* result = value;
    result->accept_(*this);
    return result;
  } else {
    return value;
  }
}
