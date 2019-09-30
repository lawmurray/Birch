/**
 * @file
 */
#if !ENABLE_LAZY_DEEP_CLONE
#include "libbirch/EagerContext.hpp"

#include "libbirch/SwapContext.hpp"

libbirch::EagerAny* libbirch::EagerContext::get(EagerAny* o) {
  auto result = m.get(o);
  return result ? result : copy(o);
}

libbirch::EagerAny* libbirch::EagerContext::copy(EagerAny* o) {
  auto alloc = static_cast<EagerAny*>(allocate(o->getSize()));
  m.put(o, alloc);
  SwapContext swapContext(this);
  return o->clone_(alloc);
}

#endif
