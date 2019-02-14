/**
 * @file
 */
#pragma once

namespace bi {
class LazyMemo;
class EagerMemo;

#if ENABLE_LAZY_DEEP_CLONE
using Memo = LazyMemo;
#else
using Memo = EagerMemo;
#endif
}
