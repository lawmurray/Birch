/**
 * @file
 */
#pragma once
#if !ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Counted.hpp"

namespace libbirch {
/**
 * Base for all class types when eager deep clone is used.
 *
 * @ingroup libbirch
 */
class EagerAny: public Counted {
public:
  using class_type_ = EagerAny;
  using this_type_ = EagerAny;

  /**
   * Constructor.
   */
  EagerAny();

  /**
   * Copy constructor.
   */
  EagerAny(const EagerAny& o);

  /**
   * Destructor.
   */
  virtual ~EagerAny();

  /**
   * Copy assignment operator.
   */
  EagerAny& operator=(const EagerAny&) = delete;

  virtual EagerAny* clone_() const {
    return new EagerAny(*this);
  }

  virtual const char* name_() const {
    return "Any";
  }
};
}

inline libbirch::EagerAny::EagerAny() :
    Counted() {
  //
}

inline libbirch::EagerAny::EagerAny(const EagerAny& o) :
    Counted(o) {
  //
}

inline libbirch::EagerAny::~EagerAny() {
  //
}

#endif
