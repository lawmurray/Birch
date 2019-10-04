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

  /**
   * Map an object that may not yet have been cloned.
   */
  EagerAny* get(EagerAny* o);

  /**
   * Deep copy.
   */
  EagerAny* copy(EagerAny* o);

  virtual EagerContext* clone_() const {
    return new LazyContext(*this);
  }

  virtual const char* name_() const {
    return "EagerContext";
  }

private:
  /**
   * Memo that maps source objects to clones.
   */
  EagerMemo m;
};
}

#endif
