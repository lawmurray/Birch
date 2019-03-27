/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyAny.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyAny::LazyAny() :
    Counted(),
    context(currentContext) {
  assert(context);
}

libbirch::LazyAny::LazyAny(const LazyAny& o) :
    Counted(o),
    context(currentContext) {
  assert(context);
}

libbirch::LazyAny::~LazyAny() {
  //
}

#endif
