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

protected:
  /**
   * Constructor.
   */
  EagerAny();

  /**
   * Destructor.
   */
  virtual ~EagerAny();

  /**
   * Copy assignment operator.
   */
  EagerAny(const EagerAny& o) = delete;
  EagerAny& operator=(const EagerAny&) = delete;

public:
  libbirch_create_function_
  libbirch_emplace_function_
  libbirch_clone_function_
  libbirch_destroy_function_

  /**
   * Name of the class.
   */
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
