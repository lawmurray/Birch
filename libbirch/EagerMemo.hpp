/**
 * @file
 */
#pragma once
#if !ENABLE_LAZY_DEEP_CLONE

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/EagerAny.hpp"
#include "libbirch/Map.hpp"

namespace bi {
/**
 * Memo for lazy deep cloning of objects.
 *
 * @ingroup libbirch
 */
class EagerMemo: public Counted {
  friend class List;
public:
  using class_type = EagerMemo;

  STANDARD_CREATE_FUNCTION
  STANDARD_EMPLACE_FUNCTION
  //STANDARD_FUNCTION
  STANDARD_DESTROY_FUNCTION

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * cloning and forwarding it if necessary.
   */
  EagerAny* get(EagerAny* o);

  /**
   * Deep copy.
   */
  EagerAny* copy(EagerAny* o);

private:
  /**
   * Memoization of source objects to clones.
   */
  Map m;
};
}

#endif
