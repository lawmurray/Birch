/**
 * @file
 */
#pragma once
#if !ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Counted.hpp"
#include "libbirch/EagerAny.hpp"
#include "libbirch/EagerMemo.hpp"

namespace libbirch {
/**
 * Context for lazy deep cloning of objects.
 *
 * @ingroup libbirch
 */
class EagerContext: public Counted {
  friend class List;
public:
  using class_type_ = EagerContext;

  libbirch_create_function_
  libbirch_emplace_function_
  libbirch_destroy_function_

  /**
   * Map an object that may not yet have been cloned.
   */
  EagerAny* get(EagerAny* o);

  /**
   * Deep copy.
   */
  EagerAny* copy(EagerAny* o);

private:
  /**
   * Memo that maps source objects to clones.
   */
  EagerMemo m;
};
}

#endif
