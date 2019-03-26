/**
 * @file
 */
#pragma once

namespace libbirch {
class LazyMemo;
class EagerMemo;

#if ENABLE_LAZY_DEEP_CLONE
using Memo = LazyMemo;
#else
using Memo = EagerMemo;
#endif
}
