/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
class LazyMemo;
class EagerMemo;

#if USE_LAZY_DEEP_CLONE
using Memo = LazyMemo;
#else
using Memo = EagerMemo;
#endif
}
