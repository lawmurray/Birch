/**
 * @file
 */
#pragma once

namespace libbirch {
class LazyContext;
class EagerContext;

/**
 * A context. This is typdefed to either LazyContext or EagerContext according
 * to the setting of the `ENABLE_LAZY_DEEP_CLONE` macro.
 */
#if ENABLE_LAZY_DEEP_CLONE
using Context = LazyContext;
#else
using Context = EagerContext;
#endif
}
