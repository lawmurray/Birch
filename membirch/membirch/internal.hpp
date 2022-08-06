/**
 * @file
 *
 * Forward declarations of types.
 */
#pragma once

namespace membirch {
template<class T> class Shared;
class Any;
class Marker;
class Scanner;
class Reacher;
class Collector;
class BiconnectedCollector;
class Spanner;
class Bridger;
class Copier;
class Memo;
class BiconnectedCopier;
class BiconnectedMemo;
class Destroyer;

/**
 * @internal
 * 
 * Flags used for bridge finding and cycle collection. For cycle collection,
 * they mostly correspond to the colors in @ref Bacon2001 "Bacon & Rajan
 * (2001)" but behave slightly differently to permit multithreading. The basic
 * principle to ensure this is that flags can be safely set during normal
 * execution (with atomic operations), but should only be unset with careful
 * consideration of thread safety.
 *
 * The flags map to colors in @ref Bacon2001 "Bacon & Rajan (2001)" as
 * follows:
 *
 *   - *buffered* maps to *purple*,
 *   - *marked* maps to *gray*,
 *   - *scanned* and *reached* together map to *black* (both on) or
 *     *white* (first on, second off),
 *   - *collected* is set once a *white* object has been destroyed.
 *
 * The use of these flags also resolves some thread safety issues that can
 * otherwise exist during the scan operation, when coloring an object white
 * (eligible for collection) then later recoloring it black (reachable); the
 * sequencing of this coloring can become problematic with multiple threads.
 */
enum Flag : int8_t {
  BUFFERED = (1 << 0),
  POSSIBLE_ROOT = (1 << 1),
  MARKED = (1 << 2),
  SCANNED = (1 << 3),
  REACHED = (1 << 4),
  COLLECTED = (1 << 5),
  CLAIMED = (1 << 6)
};

/**
 * Packed raw pointer and flag layout. Contents are:
 * 
 * - raw pointer in the first 62 bits (given alignment, the remaining two
 *   bits of the raw pointer, if represented completely, would be zero anyway),
 * - lock bit,
 * - bridge bit.
 */
enum SharedFlag : int64_t {
  BRIDGE = (1 << 0),
  LOCK = (1 << 1),
  POINTER = ~(BRIDGE|LOCK)
};

}
