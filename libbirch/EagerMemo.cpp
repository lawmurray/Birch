/**
 * @file
 */
#if !USE_LAZY_DEEP_CLONE
#include "libbirch/EagerMemo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

bi::EagerAny* bi::EagerMemo::get(EagerAny* o) {
    auto result = m.get(o);
    return result ? result : copy(o);
}

bi::EagerAny* bi::EagerMemo::copy(EagerAny* o) {
  /* for an eager deep clone we must be cautious to avoid infinite
   * recursion; memory for the new object is allocated first and put
   * in the map in case of deeper pointers back to the same object; then
   * the new object is constructed; there is no risk of another thread
   * accessing the uninitialized memory as the deep clone is not
   * accessible to other threads until completion; the new object will
   * at least have completed the EagerPtr() constructor to initialize
   * reference counts before any recursive clones occur */
  auto alloc = static_cast<EagerAny*>(allocate(o->getSize()));
  assert(alloc);
  auto uninit = m.uninitialized_put(o, alloc);
  assert(uninit == alloc);  // should be no thread contention here
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  auto result = o->clone(uninit);
  assert(result == uninit);// clone should be in the allocation
  o->incMemo();// uninitialized_put(), so responsible for ref counts
  result->incShared();
  return result;
}

#endif
