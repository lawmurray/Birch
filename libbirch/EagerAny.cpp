/**
 * @file
 */
#if !ENABLE_LAZY_DEEP_CLONE
#include "libbirch/Any.hpp"

bi::EagerAny::EagerAny() :
    Counted() {
  //
}

bi::EagerAny::EagerAny(const EagerAny& o) :
    Counted(o) {
  //
}

bi::EagerAny::~EagerAny() {
  //
}

#endif
