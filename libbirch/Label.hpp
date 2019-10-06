/**
 * @file
 */
#pragma once

namespace libbirch {
class LazyLabel;
class EagerLabel;

/**
 * A label. Either LazyLabel or EagerLabel according to the
 * `ENABLE_LAZY_DEEP_CLONE` macro.
 */
#if ENABLE_LAZY_DEEP_CLONE
using Label = LazyLabel;
#else
using Label = EagerLabel;
#endif
}
