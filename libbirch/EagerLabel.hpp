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
 * Label for bookkeeping eager deep clones.
 *
 * @ingroup libbirch
 */
class EagerLabel: public Counted {
  friend class List;
public:
  using class_type_ = EagerLabel;

  /**
   * Map an object that may not yet have been cloned.
   */
  EagerAny* get(EagerAny* o);

  /**
   * Deep copy.
   */
  EagerAny* copy(EagerAny* o);

  virtual const char* name_() const {
    return "EagerLabel";
  }

private:
  /**
   * Memo that maps source objects to clones.
   */
  EagerMemo m;
};
}

#endif
