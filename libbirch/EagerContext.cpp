/**
 * @file
 */
#if !ENABLE_LAZY_DEEP_CLONE
#include "libbirch/EagerContext.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::EagerAny* libbirch::EagerContext::get(EagerAny* o) {
    auto result = m.get(o);
    return result ? result : copy(o);
}

libbirch::EagerAny* libbirch::EagerContext::copy(EagerAny* o) {
  /* for an eager deep clone we must be cautious to avoid infinite
   * recursion; memory for the new object is allocated first and put
   * in the map in case of deeper pointers back to the same object; then
   * the new object is constructed; there is no risk of another thread
   * accessing the uninitialized memory as the deep clone is not
   * accessible to other threads until completion; the new object will
   * at least have completed the EagerPtr() constructor to initialize
   * reference counts before any recursive clones occur */
  auto alloc = static_cast<EagerAny*>(allocate(o->getSize()));
  auto uninit = alloc;
  auto singular = o->isSingular();
  if (!singular) {
    m.uninitialized_put(o, alloc);
  }
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  auto result = o->clone_(uninit);
  assert(result == uninit);  // clone should be in the allocation
  if (!singular) {
    /* uninitialized_put(), so responsible for ref counts */
    o->incMemo();
    result->incShared();
  }
  return result;
}

#endif
