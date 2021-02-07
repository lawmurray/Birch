/**
 * @file
 */
#include "libbirch/Copier.hpp"

#include "libbirch/Any.hpp"

libbirch::Any* libbirch::Copier::visit(Any* o) {
  auto& value = m.get(o);
  if (!value) {
    assert(!biconnected_copy());
    biconnected_copy(true);
    assert(biconnected_copy());
    value = o->copy();
    biconnected_copy(true);
    assert(!biconnected_copy());

    /* copy the value into a non-reference, as the reference may be
     * invalidated if m is resized during the call to accept_() below */
    Any* result = value;
    result->accept_(*this);
    return result;
  } else {
    return value;
  }
}
