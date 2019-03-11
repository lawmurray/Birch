/**
 * @file
 */
#if !ENABLE_LAZY_DEEP_CLONE
#include "libbirch/Any.hpp"

libbirch::EagerAny::EagerAny() :
    Counted() {
  //
}

libbirch::EagerAny::EagerAny(const EagerAny& o) :
    Counted(o) {
  //
}

libbirch::EagerAny::~EagerAny() {
  //
}

#endif
