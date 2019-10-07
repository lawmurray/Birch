/**
 * @file
 */
#if !ENABLE_LAZY_DEEP_CLONE
#include "libbirch/EagerLabel.hpp"

libbirch::EagerAny* libbirch::EagerLabel::get(EagerAny* o) {
  auto result = m.get(o);
  return result ? result : copy(o);
}

libbirch::EagerAny* libbirch::EagerLabel::copy(EagerAny* o) {
  auto alloc = static_cast<EagerAny*>(allocate(o->getSize()));
  m.put(o, alloc);
  return o->clone_(this, alloc);
}

#endif
